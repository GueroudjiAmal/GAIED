from __future__ import annotations

import asyncio
import logging
import os

from mol_design_agents import XTBConfig
from mol_design_agents import XTBSimulationAgent
import parsl
from parsl import Config
from parsl import HighThroughputExecutor
from parsl.concurrent import ParslPoolExecutor
from parsl.providers import LocalProvider

from academy.exchange import RedisExchangeFactory
from academy.logging import init_logging
from academy.manager import Manager

from flowcept import Flowcept

logger = logging.getLogger(__name__)


async def main() -> int:
    init_logging(logging.INFO)

    config = Config(
        executors=[
            HighThroughputExecutor(
                provider=LocalProvider(
                    worker_init=(
                        f'cd {os.getcwd()}conda activate ./mol-design;'
                    ),
                ),
                max_workers_per_node=2,
            ),
        ],
    )
    executor = ParslPoolExecutor(config)

    async with await Manager.from_exchange_factory(
        factory=RedisExchangeFactory('localhost', 6379),
        executors=executor,
    ) as manager:
        seeds = [
            'CNC(N)=O',
            'CC1=C(O)N=C(O)N1',
        ]

        agents = []
        for molecule in seeds:
            agents.append(
                await manager.launch(
                    XTBSimulationAgent,
                    args=(
                        XTBConfig(),
                        molecule,
                        20,   # max_iterations
                        3,   # no_improvement_tolerance
                    ),
                ),
            )

        print('Starting discovery campaign')
        print('=' * 80)
        # Track each agent's last known molecule count; exit when all stable.
        active: dict[int, tuple] = {i: (agent, -1) for i, agent in enumerate(agents)}
        try:
            while active:
                await asyncio.sleep(30)
                finished = []
                for i, (agent, last_count) in list(active.items()):
                    try:
                        report = await asyncio.wait_for(agent.report(), timeout=15.0)
                        count = len(report)
                        print(f'Progress report from agent {i}')
                        print(f'Number of molecules simulated: {count}')
                        print(f'Five best molecules: {report[:5]}')
                        if count == last_count and count > 0:
                            print(f'Agent {i} finished (no new molecules).')
                            finished.append(i)
                        else:
                            active[i] = (agent, count)
                    except Exception:
                        print(f'Agent {i} has finished.')
                        finished.append(i)
                for i in finished:
                    del active[i]
                print('=' * 80)
            print('All agents completed.')
        except (KeyboardInterrupt, asyncio.CancelledError):
            print('\nShutting down agents...')

    return 0


def _print_perf_summary(perf_csv: str) -> None:
    """Read the shared perf CSV and print a per-category aggregate table."""
    import csv as _csv
    from collections import defaultdict
    try:
        if not os.path.exists(perf_csv):
            return
        counts: dict = defaultdict(int)
        totals: dict = defaultdict(float)
        with open(perf_csv, newline='') as fh:
            for row in _csv.DictReader(fh):
                cat = row['category']
                counts[cat] += 1
                totals[cat] += float(row['elapsed_us'])
        if not counts:
            return
        col = 22
        header = (
            f"\n{'Category':<{col}} {'N':>7} {'Total(ms)':>11} {'Mean(µs)':>9}"
        )
        print('\n' + '=' * 60)
        print('Provenance overhead — all workers (from perf CSV):')
        print('=' * 60)
        print(header)
        print('-' * len(header.strip()))
        for cat in sorted(counts):
            n = counts[cat]
            total_us = totals[cat]
            mean_us = total_us / n if n else 0.0
            print(f"{cat:<{col}} {n:>7} {total_us/1000:>11.3f} {mean_us:>9.1f}")
    except Exception as exc:
        print(f'[perf summary] could not read {perf_csv}: {exc}')


if __name__ == '__main__':
    import signal
    import threading

    perf_csv = 'provenance_perf.csv'
    exit_code = 1
    with Flowcept() as flowcept:
        os.environ['FLOWCEPT_CAMPAIGN_ID'] = flowcept.campaign_id
        import flowcept.agents.academy.academy_plugin as _ap
        perf_csv = _ap._PERF_CSV_PATH or f'provenance_perf_{flowcept.campaign_id[:8]}.csv'
        os.environ['FLOWCEPT_PERF_CSV'] = perf_csv
        try:
            exit_code = asyncio.run(main())
        finally:
            # Suppress further Ctrl+C so cleanup isn't interrupted.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            # Run Parsl cleanup in a daemon thread; if it hangs, we still exit.
            def _parsl_cleanup():
                try:
                    parsl.dfk().cleanup()
                except Exception:
                    pass
            t = threading.Thread(target=_parsl_cleanup, daemon=True)
            t.start()
            t.join(timeout=15)
    _print_perf_summary(perf_csv)
    # os._exit bypasses atexit handlers (avoids the "DFK still running" warning).
    os._exit(exit_code)