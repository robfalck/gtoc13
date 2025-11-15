import json
from pathlib import Path

import gtoc13
from gtoc13.constants import DAY, YEAR

from gtoc13.dymos_solver.solve_arc import solve_arc


def main():
    this_dir = Path(__file__).parent

    with open(this_dir / 'bs_mission_bw25_d100_top100_20251115-015124Z_fromBody4comets.json') as f:
        bs = json.load(f)

    best = (0, len(bs['solutions'][0]['encounters']))
    for i, bs_sol in enumerate(bs['solutions']):
        N = len(bs_sol['encounters'])
        if N > best[1]:
            best = (i, N)
    print(f'best solution: {best}')

    bs_sol = bs['solutions'][best[0]]
    for i in range(2, N):
        enc_from = bs_sol['encounters'][i-1]
        enc_to = bs_sol['encounters'][i]

        from_body=enc_from['body_id']
        to_body=enc_to['body_id']

        t1 = enc_from['epoch_days'] * DAY / YEAR
        t2 = enc_to['epoch_days'] * DAY / YEAR

        print(from_body, to_body, t1, t2)

        success_i, sol_i = solve_arc(from_body=from_body, to_body=to_body,
                                     t1=t1, t2=t2,
                                     control=1,
                                     opt_dt=False,
                                     mode='feas')

        solutions_dir = this_dir

        filename = f'{i}_{from_body}_{to_body}.txt'
        solution_file = solutions_dir / f'{filename}.txt'

        sol_i.write_to_file(solution_file, precision=11)
        print(f"Solution written to {solution_file}")




if __name__ == '__main__':
    main()