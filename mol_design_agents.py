from __future__ import annotations

import asyncio
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from io import StringIO
from typing import Any
from typing import Literal

import json
import requests
import numpy as np
from ase.io import read
from ase.optimize import LBFGSLineSearch
from langchain.messages import HumanMessage
from langchain.messages import SystemMessage
from langchain.tools import tool
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import StateGraph
from rdkit import Chem
from rdkit.Chem import AllChem
from xtb.ase.calculator import XTB

from academy.agent import action
from academy.agent import Agent
from academy.agent import loop

logger = logging.getLogger(__name__)

_SomeAPI_URL = "https://"
_SomeAPI_USER = os.environ.get("SomeAPI_USER", "")


class _SomeAPIResponse:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class SomeAPILLM:
    """Drop-in replacement for ChatOpenAI that routes calls through the SomeAPI gateway."""

    def __init__(self, model: str, temperature: float = 0.1) -> None:
        self.model = model
        self.temperature = temperature
        self._tools: list | None = None

    def bind_tools(self, tools: list) -> SomeAPILLM:
        bound = SomeAPILLM(self.model, self.temperature)
        bound._tools = tools
        return bound

    async def ainvoke(self, messages: list) -> _SomeAPIResponse:
        system_content = ""
        user_content = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_content = msg.content
            elif isinstance(msg, HumanMessage):
                user_content = msg.content

        if self._tools:
            tool_names = ", ".join(t.name for t in self._tools)
            user_content = (
                f"{user_content}\n\n"
                f"Available tools: {tool_names}.\n"
                'Respond with ONLY a JSON object: '
                '{"tool_calls": [{"name": "<tool>", "args": {"smiles": "<SMILES>"}}]}'
            )

        is_claude = self.model.lower().startswith("claude")
        if is_claude:
            payload = {
                "user": _SomeAPI_USER,
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                "stop": [],
                "temperature": self.temperature,
            }
        else:
            payload = {
                "user": _SomeAPI_USER,
                "model": self.model,
                "system": system_content,
                "prompt": [user_content],
                "stop": [],
                "temperature": self.temperature,
                "top_p": 0.9,
            }

        def _post() -> dict:
            resp = requests.post(
                _SomeAPI_URL,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()

        data = await asyncio.to_thread(_post)
        content = data["response"]

        if self._tools:
            try:
                start = content.find('{')
                end = content.rfind('}')
                parsed = json.loads(content[start:end + 1])
                tool_calls = parsed.get('tool_calls', [])
            except (json.JSONDecodeError, ValueError):
                tool_calls = []
            return _SomeAPIResponse(content=content, tool_calls=tool_calls)

        return _SomeAPIResponse(content=content)


def _generate_initial_xyz(mol_string: str) -> str:
    """Generate the XYZ coordinates for a molecule.

    Args:
        mol_string: SMILES string

    Returns:
        - InChI string for the molecule
        - XYZ coordinates for the molecule
    """

    # Generate 3D coordinates for the molecule
    mol = Chem.MolFromSmiles(mol_string)
    if mol is None:
        raise ValueError(f'Parse failure for {mol_string}')
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)

    # Save geometry as 3D coordinates
    xyz = f'{mol.GetNumAtoms()}\n'
    xyz += mol_string + '\n'
    conf = mol.GetConformer()
    for i, a in enumerate(mol.GetAtoms()):
        s = a.GetSymbol()
        c = conf.GetAtomPosition(i)
        xyz += f'{s} {c[0]} {c[1]} {c[2]}\n'

    return xyz


@dataclass
class XTBConfig:
    accuracy: float = 0.05
    search_fmax: float = 0.02
    search_steps: int = 250


def _compute_energy_sync(config: XTBConfig, smiles: str) -> float:
    """Run the ionization potential computation."""

    # Make the initial geometry
    xyz = _generate_initial_xyz(smiles)

    # Make the XTB calculator
    calc = XTB(accuracy=config.accuracy)

    # Parse the molecule
    atoms = read(StringIO(xyz), format='xyz')

    # Compute the neutral geometry
    # Uses QCEngine (https://github.com/MolSSI/QCEngine)
    # to handle interfaces to XTB
    atoms.calc = calc
    dyn = LBFGSLineSearch(atoms, logfile=None)
    dyn.run(fmax=config.search_fmax, steps=config.search_steps)

    neutral_energy = atoms.get_potential_energy()

    # Compute the energy of the relaxed geometry in charged form
    charges = np.ones((len(atoms),)) * (1 / len(atoms))
    atoms.set_initial_charges(charges)
    charged_energy = atoms.get_potential_energy()

    return charged_energy - neutral_energy


def _compute_energy_with_prov(
    config: XTBConfig,
    smiles: str,
    parent_task_id: str | None,
    workflow_id: str | None,
    campaign_id: str | None,
) -> float:
    """Wrapper around _compute_energy_sync that records provenance in the XTB worker.

    Runs inside the ProcessPoolExecutor worker process.  _worker_init (injected
    by the patched ProcessPoolExecutor.__init__) has already set up
    _ap._ACTIVE_INTERCEPTOR in this process, so we just call intercept_task()
    after the computation finishes to capture inputs, outputs, timing, and the
    link back to the Academy action that dispatched us (parent_task_id).
    """
    import time as _time

    started_at = _time.time()
    result = None
    error = None
    try:
        result = _compute_energy_sync(config, smiles)
        return result
    except Exception as exc:
        error = exc
        raise
    finally:
        try:
            import flowcept.agents.academy.academy_plugin as _ap
            interceptor = _ap._ACTIVE_INTERCEPTOR
            if interceptor is not None:
                task: dict = {
                    'activity_id': 'compute_ionization_energy',
                    'subtype': 'parsl_task',
                    'status': 'ERROR' if error else 'FINISHED',
                    'started_at': started_at,
                    'ended_at': _time.time(),
                    'used': {'smiles': smiles},
                    'generated': (
                        {'ionization_energy': result} if result is not None else None
                    ),
                    'stderr': str(error) if error else None,
                }
                if parent_task_id:
                    task['parent_task_id'] = parent_task_id
                if workflow_id:
                    task['workflow_id'] = workflow_id
                if campaign_id:
                    task['campaign_id'] = campaign_id
                interceptor.intercept_task(task)
        except Exception:
            pass


@dataclass
class SearchState:
    seed: str
    plan: str = ''
    tool_calls: list[Any] = field(default_factory=list)
    simulated_molecules: dict[str, float] = field(default_factory=dict)
    conclusions: list[str] = field(default_factory=list)
    critique: str = ''


class XTBSimulationAgent(Agent):
    """Agent for running XTB to characterize molecules."""

    def __init__(
        self,
        xtb_config: XTBConfig,
        start_molecule: str,
        max_iterations: int | None = None,
        no_improvement_tolerance: int | None = None,
        reasoning_model: str = 'gpt54',
        generation_model: str = 'gpt54',
    ):
        self.config = xtb_config
        self.reasoning_model = reasoning_model
        self.generation_model = generation_model
        self.search_state = SearchState(seed=start_molecule)
        self.molecule_cache: dict[str, float] = {}
        self.max_iterations = max_iterations
        self.no_improvement_tolerance = no_improvement_tolerance

    async def agent_on_startup(self) -> None:
        n_workers: int
        if sys.platform == 'darwin':  # pragma: no cover
            n_workers = 2
        else:
            n_workers = max(
                len(os.sched_getaffinity(0)) - 1,
                1,
            )  # Get cores we are assigned to

        # ── Flowcept provenance (optional) ──────────────────────────────────
        # FLOWCEPT_CAMPAIGN_ID is set by run_07.py and inherited by workers.
        # Interceptors are started BEFORE pool creation so that the patched
        # ProcessPoolExecutor.__init__ (from the main process) sees a live
        # _ACTIVE_INTERCEPTOR and injects _worker_init into XTB pool workers,
        # enabling Parsl-level task provenance capture.
        self._academy_interceptor = None
        self._lg_interceptor = None
        self._lg_callbacks: list = []
        campaign_id = os.environ.get('FLOWCEPT_CAMPAIGN_ID')
        if campaign_id:
            # ── Block 1: interceptors + SomeAPILLM patch (critical path) ──────
            # This block must succeed for any provenance to work.  It does NOT
            # import langchain_core so it is independent of handler availability.
            try:
                import time as _time
                import flowcept.agents.academy.academy_plugin as _ap
                from flowcept.agents.academy.academy_plugin import (
                    AcademyInterceptor,
                    _install_runtime_patches,
                )
                self._academy_interceptor = AcademyInterceptor()
                self._academy_interceptor.start(
                    f'xtb-agent-{self.search_state.seed}',
                    campaign_id=campaign_id,
                )
                _ap._ACTIVE_INTERCEPTOR = self._academy_interceptor
                _install_runtime_patches()

                import flowcept.agents.langgraph.langgraph_plugin as _lgp
                from flowcept.agents.langgraph.langgraph_plugin import (
                    LangGraphInterceptor,
                    _ProvenanceStats,
                )
                self._lg_interceptor = LangGraphInterceptor()
                self._lg_interceptor.start(
                    f'xtb-graph-{self.search_state.seed}',
                    campaign_id=campaign_id,
                )
                self._prov_stats = _ProvenanceStats()
                _lgp._ACTIVE_INTERCEPTOR = self._lg_interceptor
                _lgp._PROV_STATS = self._prov_stats
                # Route academy-level overhead (action_emit, loop_emit, flush)
                # into the same stats object for a comprehensive report.
                _ap._PROV_STATS = self._prov_stats

                # Patch SomeAPILLM here, before handler build, so LLM calls are
                # always traced even if langchain_core is unavailable.
                _orig_ainvoke = SomeAPILLM.ainvoke
                _interceptor_ref = self._lg_interceptor
                _stats_ref = self._prov_stats

                async def _traced_ainvoke(self_llm, messages):
                    system_content = ''
                    user_content = ''
                    for msg in messages:
                        if isinstance(msg, SystemMessage):
                            system_content = msg.content
                        elif isinstance(msg, HumanMessage):
                            user_content = msg.content

                    t_llm = _time.perf_counter()
                    result = await _orig_ainvoke(self_llm, messages)
                    elapsed_ms = round((_time.perf_counter() - t_llm) * 1000, 2)

                    t_prov = _time.perf_counter()
                    try:
                        _interceptor_ref.intercept_task({
                            'activity_id': f'SomeAPI_llm/{self_llm.model}',
                            'subtype': 'llm_call',
                            'status': 'FINISHED',
                            'used': {
                                'model': self_llm.model,
                                'temperature': self_llm.temperature,
                                'has_tools': bool(self_llm._tools),
                                'system_prompt': system_content,
                                'user_prompt': user_content,
                            },
                            'generated': {
                                'response': result.content,
                                'tool_calls': result.tool_calls,
                            },
                            'custom_metadata': {'elapsed_ms': elapsed_ms},
                        })
                    except Exception:
                        pass
                    _stats_ref.record('llm_intercept', _time.perf_counter() - t_prov)
                    return result

                SomeAPILLM.ainvoke = _traced_ainvoke
                logger.info('Flowcept provenance enabled for this worker.')
            except Exception as exc:
                logger.warning('Flowcept init failed in worker: %s', exc)
                self._academy_interceptor = None
                self._lg_interceptor = None

            # ── Block 2: LangGraph callback handler (optional) ─────────────
            # Depends on langchain_core.  Failure here does not affect the
            # SomeAPILLM patch or academy-level provenance set up above.
            if self._lg_interceptor is not None:
                try:
                    from flowcept.agents.langgraph.langgraph_plugin import _build_handler_class
                    _HandlerCls = _build_handler_class()
                    self._lg_callbacks = [_HandlerCls(self._lg_interceptor, self._prov_stats)]
                except Exception as exc:
                    logger.warning('LangGraph callback handler unavailable: %s', exc)

        # Pool is created AFTER _ap._ACTIVE_INTERCEPTOR is set so that the
        # patched ProcessPoolExecutor.__init__ (from the main process) injects
        # _worker_init into XTB pool workers, wiring up per-task provenance.
        self.pool = ProcessPoolExecutor(max_workers=n_workers)

        tools = [tool(self.compute_ionization_energy)]
        self.reasoning_llm = SomeAPILLM(model=self.reasoning_model)
        self.generation_llm = SomeAPILLM(model=self.generation_model)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.llm_with_tools = self.generation_llm.bind_tools(tools)

    async def agent_on_shutdown(self) -> None:
        if self._lg_interceptor is not None:
            try:
                self._lg_interceptor.stop()
            except Exception as exc:
                logger.warning('Flowcept LangGraph interceptor stop failed: %s', exc)
        if self._academy_interceptor is not None:
            try:
                self._academy_interceptor.stop()
            except Exception as exc:
                logger.warning('Flowcept Academy interceptor stop failed: %s', exc)
        if getattr(self, '_prov_stats', None) is not None:
            perf_csv = os.environ.get('FLOWCEPT_PERF_CSV')
            if perf_csv:
                try:
                    wf_id = getattr(self._lg_interceptor, '_workflow_id', '') or ''
                    self._prov_stats.to_csv(perf_csv, workflow_id=wf_id)
                except Exception as exc:
                    logger.warning('Flowcept perf stats write failed: %s', exc)
        self.pool.shutdown()

    async def compute_ionization_energy(self, smiles: str) -> float:
        """Compute the ionization energy for the given molecule.

        Args:
            smiles: SMILES string to evaluate
        Returns:
            Ionization energy in Ha
        """
        if smiles in self.molecule_cache:
            return self.molecule_cache[smiles]

        # Capture the current Academy action task_id so the Parsl task can be
        # linked back to it via parent_task_id in the provenance graph.
        parent_task_id = None
        workflow_id = None
        campaign_id = None
        if self._academy_interceptor is not None:
            try:
                from flowcept.agents.academy.academy_plugin import _current_action_task_id
                parent_task_id = _current_action_task_id.get(None)
            except Exception:
                pass
            workflow_id = self._academy_interceptor._workflow_id
            campaign_id = self._academy_interceptor._campaign_id

        loop = asyncio.get_event_loop()
        ionization_energy = await loop.run_in_executor(
            self.pool,
            _compute_energy_with_prov,
            self.config,
            smiles,
            parent_task_id,
            workflow_id,
            campaign_id,
        )
        self.molecule_cache[smiles] = ionization_energy
        return ionization_energy

    @loop
    async def conduct_simulation_campaign(  # noqa: C901,PLR0915
        self,
        shutdown: asyncio.Event,
    ) -> None:
        """Conduct a simulation campaign.

        This loop uses an LLM to conduct a search for molecules with high
        ionizationenergy, using the starting molecule as a seed for the
        search space.
        """

        async def plan(state: SearchState) -> SearchState:
            response = await self.reasoning_llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            'You are a expert computational chemist tasked '
                            'with finding molecules with desired properties. '
                            'You should use simulations to explore the '
                            'chemical space in search of better molecules. '
                            'Come up with a plan for novel molecules to '
                            'simulate. Ground your proposed molecules in '
                            'knowledge of chemical and physical properties. '
                            'For each proposed molecule that you simulate, '
                            'provide the reasoning about why you expect this '
                            'molecule to be better either than previous '
                            'molecules that you have seen, or what you hope '
                            'to learn from running the simulation. In addition'
                            ' to the target property, you should always '
                            'consider things like cost and sythesizability. '
                            'Note: the only experiment your lab is capable of '
                            'running is calculating the ionization energy of '
                            'proposed molecules.'
                        ),
                    ),
                    HumanMessage(
                        content=(
                            'Look for molecules with high ionization energy '
                            f'starting with the molecule {state.seed}'
                        ),
                    ),
                ],
            )

            logger.info(f'Planner: {response}')

            return SearchState(
                seed=state.seed,
                plan=response.content,
                tool_calls=state.tool_calls,
                simulated_molecules=state.simulated_molecules,
                conclusions=state.conclusions,
                critique=state.critique,
            )

        async def tool_calling(state: SearchState) -> SearchState:
            previously_simulated = ' '.join(
                state.simulated_molecules.keys(),
            )
            response = await self.llm_with_tools.ainvoke(
                [
                    SystemMessage(
                        content=(
                            'You are a computational chemist who is an expert '
                            'at conducting simulations. Using proposed plan '
                            'and chemical reasoning provided, decide which '
                            'simulations to run using the given tool. Do not '
                            'write python code. You should run several '
                            'simulations at once. Do not repeat previously '
                            'conducted simulations.'
                        ),
                    ),
                    HumanMessage(
                        content=(
                            f'Plan: {state.plan}\n\n'
                            f'Simulated Molecules: {previously_simulated} '
                            'Do not repeat those molecules in the tool calls.'
                        ),
                    ),
                ],
            )
            logger.info(f'Tool Calling: {response}')
            while len(response.tool_calls) == 0:
                logger.warning('Tool calling agent did not call tools!')
                logger.warning(response)
                response = await self.llm_with_tools.ainvoke(
                    [
                        SystemMessage(
                            content=(
                                'You are a computational chemist who is '
                                'an expert at conducting simulations. '
                                'Using proposed plan and chemical reasoning'
                                ' provided, decide which simulations to run'
                                ' using the given tool. Do not write python '
                                'code. You should run several simulations at '
                                'once. Do not repeat previously conducted '
                                'simulations.'
                            ),
                        ),
                        HumanMessage(
                            content=(
                                f'Plan: {state.plan}\n\n'
                                f'Simulated Molecules: {previously_simulated} '
                                'Do not repeat those molecules in the tool '
                                'calls.'
                            ),
                        ),
                        response,
                        HumanMessage(
                            content=(
                                'That response does not contain any correctly'
                                'formatted tool calls that could be parsed. '
                                'Please try again! Do not write python code!'
                            ),
                        ),
                    ],
                )

            return SearchState(
                seed=state.seed,
                plan=state.plan,
                tool_calls=response.tool_calls,
                simulated_molecules=state.simulated_molecules,
                conclusions=state.conclusions,
                critique=state.critique,
            )

        async def simulate(state: SearchState) -> SearchState:
            """Performs the tool call"""
            logger.warning(
                f'Previous Simulations: {len(state.simulated_molecules)}',
            )
            logger.warning(f'New Simulations: {len(state.tool_calls)}')
            if len(state.tool_calls) == 0:
                return state

            task_to_smiles = {}
            for tool_call in state.tool_calls:
                tool = self.tools_by_name[tool_call['name']]
                task = asyncio.create_task(tool.ainvoke(tool_call['args']))
                task_to_smiles[task] = tool_call['args']['smiles']

            done, _ = await asyncio.wait(task_to_smiles)

            results = {}
            for task in done:
                smiles = task_to_smiles[task]
                try:
                    results[smiles] = await task
                except Exception as e:
                    results[smiles] = str(e)

            new_results = state.simulated_molecules | results

            return SearchState(
                seed=state.seed,
                plan=state.plan,
                tool_calls=[],
                simulated_molecules=new_results,
                conclusions=state.conclusions,
                critique=state.critique,
            )

        async def conclude(state: SearchState) -> SearchState:
            results_str = '\n'.join(
                f'{molecule}\t\t{energy}'
                for molecule, energy in state.simulated_molecules.items()
            )
            response = await self.generation_llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            'You are expert chemist analyzing the results'
                            'of simulations. Synthesize any conclusions from'
                            ' the planned computational campaign and the '
                            'simulation results so far. A -inf result means '
                            'the simulation could not be run, likely because '
                            'the SMILES string of the molecule was invalid. '
                            'You should not repeat conclusions that have '
                            'already been reached but should correct the '
                            'conclusions if they are wrong. For each '
                            'conclusion, output a concise, stand-alone '
                            'statement. Return only the new conclusions.'
                        ),
                    ),
                    HumanMessage(
                        content=(
                            f'Plan: {state.plan}\n\n'
                            'Simulation Results:\n'
                            'Molecule\t\tEnergy\n'
                            f'{results_str}'
                        ),
                    ),
                ],
            )
            logger.info(f'Conclusions: {response}')
            new_state = SearchState(
                seed=state.seed,
                plan=response.content,
                tool_calls=state.tool_calls,
                simulated_molecules=state.simulated_molecules,
                conclusions=[*state.conclusions, response.content],
                critique=state.critique,
            )
            self.search_state = new_state
            return new_state

        async def critique(state: SearchState) -> SearchState:
            results_str = '\n'.join(
                f'{molecule}\t\t{energy}'
                for molecule, energy in state.simulated_molecules.items()
            )
            conclusions = '\n\t'.join(state.conclusions)
            response = await self.generation_llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            'You are expert chemist analyzing the results of '
                            'simulations. Critique the planned simulation '
                            'campaign based on the chemistry foundations, '
                            'the computational results, and the conclusions '
                            'drawn. Be extremely critical. Look for flaws in '
                            'the methodology and reasoning.'
                        ),
                    ),
                    HumanMessage(
                        content=(
                            f'Plan: {state.plan}\n\n'
                            'Simulation Results:\n'
                            'Molecule\t\tEnergy\n'
                            f'{results_str}\n\n'
                            'Conclusions:\n'
                            f'\t{conclusions}'
                        ),
                    ),
                ],
            )
            logger.info(f'Critique: {response}')
            return SearchState(
                seed=state.seed,
                plan=state.plan,
                tool_calls=state.tool_calls,
                simulated_molecules=state.simulated_molecules,
                conclusions=state.conclusions,
                critique=response.content,
            )

        async def update(state: SearchState) -> SearchState:
            results_str = '\n'.join(
                f'{molecule}\t\t{energy}'
                for molecule, energy in state.simulated_molecules.items()
            )
            conclusions = '\n\t'.join(state.conclusions)
            response = await self.reasoning_llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            'You are expert chemist analyzing the results of '
                            'simulations. Update the given plan based on the '
                            'results so far, the conclusions drawn and the '
                            'critique given. Ground your updates in current '
                            'results, but also knowledge of chemistry and '
                            'physics. You should output a new, stand-alone '
                            'plan that does not make reference to the '
                            'existing plan. If all of the experiments '
                            'suggested by the plan have been carried out, '
                            'suggest more experiments. Note: the only '
                            'experiment your lab is capable of running is '
                            'calculating the ionization energy of proposed '
                            'molecules.'
                        ),
                    ),
                    HumanMessage(
                        content=(
                            f'Plan: {state.plan}\n\n'
                            'Simulation Results:\n'
                            'Molecule\t\tEnergy\n'
                            f'{results_str}\n\n'
                            'Conclusions:\n'
                            f'\t{conclusions}\n\n'
                            'Critique:\n'
                            f'\t{state.critique}'
                        ),
                    ),
                ],
            )
            logger.info(f'Update: {response}')

            return SearchState(
                seed=state.seed,
                plan=response.content,
                tool_calls=state.tool_calls,
                simulated_molecules=state.simulated_molecules,
                conclusions=state.conclusions,
                critique=state.critique,
            )

        _iteration = 0
        _no_improve_count = 0
        _best_energy: float = float('-inf')

        async def should_continue(
            state: SearchState,
        ) -> Literal['critique', END]:  # type: ignore[valid-type]
            nonlocal _iteration, _no_improve_count, _best_energy
            _iteration += 1

            # Track whether the global best energy improved this round.
            # Filter out failed simulations stored as strings.
            numeric = [
                v for v in state.simulated_molecules.values()
                if isinstance(v, (int, float))
            ]
            if numeric:
                current_best = max(numeric)
                if current_best > _best_energy:
                    _best_energy = current_best
                    _no_improve_count = 0
                else:
                    _no_improve_count += 1

            if shutdown.is_set():
                return END
            if self.max_iterations is not None and _iteration >= self.max_iterations:
                logger.info(
                    'Stopping after %d iterations (max_iterations=%d)',
                    _iteration, self.max_iterations,
                )
                shutdown.set()
                return END
            if (
                self.no_improvement_tolerance is not None
                and _no_improve_count >= self.no_improvement_tolerance
            ):
                logger.info(
                    'Stopping: no improvement for %d consecutive iterations',
                    _no_improve_count,
                )
                shutdown.set()
                return END
            return 'critique'

        # Build workflow
        agent_builder = StateGraph(SearchState)

        # Add nodes
        agent_builder.add_node('plan', plan)
        agent_builder.add_node('tool_calling', tool_calling)
        agent_builder.add_node('simulate', simulate)
        agent_builder.add_node('conclude', conclude)
        agent_builder.add_node('critique', critique)
        agent_builder.add_node('update', update)

        # Add edges to connect nodes
        agent_builder.add_edge(START, 'plan')
        agent_builder.add_edge('plan', 'tool_calling')
        agent_builder.add_edge('tool_calling', 'simulate')
        agent_builder.add_edge('simulate', 'conclude')
        agent_builder.add_conditional_edges(
            'conclude',
            should_continue,
            ['critique', END],
        )
        agent_builder.add_edge('critique', 'update')
        agent_builder.add_edge('update', 'tool_calling')

        # Compile the agent
        agent = agent_builder.compile()

        # Run until the agent is terminated
        await agent.ainvoke(
            self.search_state,
            {'recursion_limit': 10000, 'callbacks': self._lg_callbacks},
        )

    @action
    async def report(self) -> list[tuple[str, float]]:
        """Summarize findings so far."""
        best_molecules = sorted(
            self.molecule_cache.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return best_molecules

