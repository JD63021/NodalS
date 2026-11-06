#!/usr/bin/env python3
"""Case-driven restart/rebuild scheduling for the existing NodalS face-y+ patch.

No CUDA source changes, no mesh-name gates, no settings-lock file.
Rerunning without --resume overwrites the runner's existing stage directories.
"""
from __future__ import annotations
import argparse
import configparser
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime

STAGE_RE = re.compile(r'^stage_\d{4,}_\d{4,}$')


def parse_bool(v):
    if str(v).strip().lower() in ('yes', 'true', 'on', '1'):
        return True
    if str(v).strip().lower() in ('no', 'false', 'off', '0'):
        return False
    raise ValueError(f'Expected true/false; got {v!r}')


def resolve_path(v, relative_to):
    p = Path(os.path.expandvars(os.path.expanduser(str(v))))
    return (p if p.is_absolute() else relative_to / p).resolve()


def read_config(case, overrides):
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#', ';'), strict=True)
    with case.open() as f:
        cp.read_file(f)
    for item in overrides:
        try:
            key, value = item.split('=', 1)
            section, option = key.rsplit('.', 1)
        except ValueError as ex:
            raise ValueError(f'Expected --set SECTION.KEY=VALUE, got {item!r}') from ex
        if not cp.has_section(section):
            cp.add_section(section)
        cp.set(section, option, value)
    return cp


def pick(cp, section, key, fallback):
    return cp.get(section, key, fallback=str(fallback))


def schedule(final_iteration, enabled, first, every, limit):
    if final_iteration < 1 or first < 1 or every < 1 or limit < -1:
        raise ValueError('final iterations, first_refresh and refresh_every must be positive; max_refreshes >= -1')
    refreshes = []
    if enabled and limit != 0:
        n = first
        while n < final_iteration and (limit == -1 or len(refreshes) < limit):
            refreshes.append(n)
            n += every
    boundaries = [0, *refreshes, final_iteration]
    return refreshes, [(b, e) for b, e in zip(boundaries[:-1], boundaries[1:])]


def run():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', type=Path, required=True, help='Original NodalS .case plus [adaptive_sampling] section')
    ap.add_argument('--root', type=Path, default=Path('~/Downloads/NodalS_SST_SOFTWARE_V2'))
    ap.add_argument('--mesh', type=Path, help='Override [mesh] path; any compatible tetrahedral .msh/HXT1')
    ap.add_argument('--output', type=Path, help='Override [output] directory')
    ap.add_argument('--final-iterations', type=int, help='Override [simple] max_iterations')
    ap.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE', help='Change any case value for this run')
    ap.add_argument('--resume', action='store_true', help='Skip complete stages; rerun incomplete stages')
    ap.add_argument('--dry-run', action='store_true', help='Print complete plan and native-wrapper commands without executing')
    a = ap.parse_args()
    case = resolve_path(a.case, Path.cwd())
    root = resolve_path(a.root, Path.cwd())
    if not case.is_file():
        raise FileNotFoundError(case)
    driver = root / 'nodals.py'
    if not driver.is_file():
        raise FileNotFoundError(f'NodalS executable front end missing: {driver}')
    cp = read_config(case, a.set)
    final = a.final_iterations if a.final_iterations is not None else int(pick(cp, 'simple', 'max_iterations', '1000'))
    active = parse_bool(pick(cp, 'adaptive_sampling', 'enabled', 'true'))
    first = int(pick(cp, 'adaptive_sampling', 'first_refresh', '100'))
    every = int(pick(cp, 'adaptive_sampling', 'refresh_every', '100'))
    count = int(pick(cp, 'adaptive_sampling', 'max_refreshes', '-1'))
    target_sp = float(pick(cp, 'adaptive_sampling', 'target_spalding_yplus', '30'))
    target_om = float(pick(cp, 'adaptive_sampling', 'target_omega_yplus', '30'))
    minimum = float(pick(cp, 'adaptive_sampling', 'min_scale', '0.02'))
    maximum = float(pick(cp, 'adaptive_sampling', 'max_scale', '0.35'))
    if not (0 < target_sp <= 500 and 0 < target_om <= 500 and 0 < minimum <= maximum <= 1):
        raise ValueError('Invalid target y+ or sampling-scale bounds')
    refreshes, stages = schedule(final, active, first, every, count)
    output_string = a.output or pick(cp, 'output', 'directory', 'runs/ubend_adaptive')
    out = resolve_path(output_string, Path.cwd() if a.output else case.parent)
    mesh = resolve_path(a.mesh, Path.cwd()) if a.mesh else None
    source_mesh = mesh or resolve_path(pick(cp, 'mesh', 'path', ''), case.parent)
    if not source_mesh.is_file():
        raise FileNotFoundError('Mesh missing: ' + str(source_mesh))
    # Only physical prerequisites for the current face-y+ implementation.
    if refreshes:
        if (int(pick(cp, 'wall_function', 'samples', '1')) != 1 or
                int(pick(cp, 'omega_wall', 'sample_index', '3')) != 3 or
                pick(cp, 'wall_sampling', 'mode', 'ray_trace').lower() != 'ray_trace' or
                not parse_bool(pick(cp, 'wall_function', 'face_spalding', 'true'))):
            raise ValueError('Current face-y+ patch requires wall_function.samples=1, omega_wall.sample_index=3, wall_sampling.mode=ray_trace, wall_function.face_spalding=true')
    if out in {Path('/'), Path.home().resolve(), root, root.parent}:
        raise ValueError('Output directory must be a dedicated run directory, not home/root/source')
    print('NODALS_ADAPTIVE_CASE=' + str(case), flush=True)
    print('NODALS_ADAPTIVE_MESH=' + str(source_mesh), flush=True)
    print('NODALS_ADAPTIVE_OUTPUT=' + str(out), flush=True)
    print(f'NODALS_ADAPTIVE_PLAN final={final} first={first} every={every} maxRefreshes={count} refreshAt={refreshes} spYplus={target_sp:g} omegaYplus={target_om:g}', flush=True)
    for index, (begin, end) in enumerate(stages):
        print(f'NODALS_ADAPTIVE_STAGE index={index} from={begin} to={end} sampling={"fixed" if index==0 else "target_yplus"} refreshCount={index}', flush=True)

    # Standard .case values are passed through, not replaced with a hard-coded
    # set of solver parameters. Only segment checkpoint/ray controls are injected.
    common = [sys.executable, '-u', str(driver), 'run', '--case', str(case)]
    if mesh:
        common += ['--mesh', str(mesh)]
    for override in a.set:
        common += ['--set', override]
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        # No settings identity gate: freely re-run with any case, parameters and
        # supported mesh. Only managed stage directories are overwritten.
        if not a.resume:
            for child in out.iterdir():
                if child.is_dir() and STAGE_RE.fullmatch(child.name):
                    shutil.rmtree(child)
    manifest = {'case': str(case), 'mesh': str(source_mesh), 'root': str(root),
                'refreshes': refreshes, 'stages': stages, 'final_iterations': final,
                'target_sp_yplus': target_sp, 'target_omega_yplus': target_om,
                'started_utc': datetime.now().isoformat(),
                'overrides': a.set, 'completed': []}
    if not a.dry_run:
        (out/'adaptive_schedule.json').write_text(json.dumps(manifest, indent=2) + '\n')
    prev = None
    for index, (begin, end) in enumerate(stages):
        segment = out / f'stage_{begin:04d}_{end:04d}'
        checkpoint = segment / f'NODALS_STAGE31_STATE_it{end}.ckpt'
        faceout = segment / f'NODALS_FACE_UTAU_it{end}.csv'
        if a.resume and not a.dry_run and checkpoint.is_file() and checkpoint.stat().st_size > 0 and (not refreshes or (faceout.is_file() and faceout.stat().st_size > 0)):
            print(f'NODALS_ADAPTIVE_SKIP_COMPLETE from={begin} to={end}', flush=True)
            prev = segment
            manifest['completed'].append(end)
            continue
        if not a.dry_run and segment.exists():
            shutil.rmtree(segment)
        # End-of-segment save must occur at an absolute SIMPLE iteration end.
        # The requested cadence is preserved if it divides that iteration;
        # otherwise use the end iteration to force the required checkpoint.
        user_ck = int(pick(cp, 'output', 'checkpoint_every', '100'))
        user_vtu = int(pick(cp, 'output', 'vtu_every', '100'))
        checkpoint_every = user_ck if user_ck > 0 and end % user_ck == 0 else end
        vtu_every = user_vtu if user_vtu > 0 and end % user_vtu == 0 else end
        cmd = common + [
            '--set', f'output.checkpoint_every={checkpoint_every}',
            '--set', f'output.vtu_every={vtu_every}',
            '--set', 'output.checkpoint_dir=.',
            '--set', 'output.vtu_dir=.',
            '--set', 'environment.NODALS_FACE_YPLUS_EXPORT=' + ('1' if refreshes else '0'),
            '--set', f'environment.NODALS_FACE_YPLUS_SP={target_sp:.17g}',
            '--set', f'environment.NODALS_FACE_YPLUS_OMEGA={target_om:.17g}',
            '--set', f'environment.NODALS_FACE_YPLUS_MIN_SCALE={minimum:.17g}',
            '--set', f'environment.NODALS_FACE_YPLUS_MAX_SCALE={maximum:.17g}',
            '--set', 'restart.path=',
        ]
        if index > 0:
            assert prev is not None
            input_ck = prev / f'NODALS_STAGE31_STATE_it{begin}.ckpt'
            input_face = prev / f'NODALS_FACE_UTAU_it{begin}.csv'
            if not a.dry_run and (not input_ck.is_file() or not input_face.is_file()):
                raise RuntimeError(f'Missing stage inputs: {input_ck} / {input_face}')
            cmd += ['--restart', str(input_ck),
                    '--set', f'environment.NODALS_FACE_YPLUS_UTAU_CSV={input_face}']
        else:
            # Avoid stale adaptive sample selection on a fresh run.
            cmd += ['--set', 'environment.NODALS_FACE_YPLUS_UTAU_CSV=']
        cmd += ['--steps', str(end-begin), '--output', str(segment)]
        if a.dry_run:
            print(f'NODALS_ADAPTIVE_DRY_CMD from={begin} to={end}: {shlex.join(cmd)}', flush=True)
            prev = segment
            continue
        print(f'NODALS_ADAPTIVE_START from={begin} to={end} refresh={index} cmd={shlex.join(cmd)}', flush=True)
        # nodals.py streams native output. Exit 3 is normal when requested
        # number of iterations ended without formal convergence.
        rc = subprocess.call(cmd, cwd=root)
        if rc not in (0, 3):
            raise RuntimeError(f'NodalS segment {begin}->{end} exited {rc}; see {segment}/terminal.log')
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError(f'Segment {begin}->{end} did not save checkpoint {checkpoint}')
        if refreshes and (not faceout.is_file() or faceout.stat().st_size == 0):
            raise RuntimeError(f'Segment {begin}->{end} did not export F2 wall u_tau {faceout}')
        if index > 0:
            terminal_log = (segment / 'terminal.log').read_text(errors='replace')
            if 'STAGE31_RESTART_LOAD=PASS' not in terminal_log or not re.search(r'NODALS_FACE_YPLUS_LOADED[^\n]*status=PASS', terminal_log):
                raise RuntimeError(f'Segment {begin}->{end} did not verify both checkpoint restart and adaptive ray loading')
        prev = segment
        manifest['completed'].append(end)
        (out/'adaptive_schedule.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'NODALS_ADAPTIVE_PASS from={begin} to={end} checkpoint={checkpoint} faceUTau={faceout}', flush=True)
    print(f'NODALS_ADAPTIVE_{"DRY_RUN" if a.dry_run else "COMPLETE"} iteration={final} output={out}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(run())
    except (OSError, ValueError, RuntimeError, configparser.Error) as exc:
        print('NODALS_ADAPTIVE_ERROR: ' + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(2)
