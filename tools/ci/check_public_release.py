"""Block accidental publication of local models/data, including Git history.

This path-based guard supplements .gitignore; it is not a legal clearance or a
secret/content scanner. Existing upstream assets remain separately licensed.
"""
import argparse
import fnmatch
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = ROOT / 'tools/ci/public_asset_allowlist.txt'
PAYLOADS = {'.pkl','.pickle','.npz','.npy','.pt','.pth','.ckpt','.safetensors',
            '.onnx','.h5','.hdf5','.obj','.ply','.stl','.fbx','.usd','.usda',
            '.usdc','.usdz','.glb','.gltf','.blend','.zip','.tar','.tgz','.7z','.rar','.crdownload'}
LOCAL_ROOTS = {'out','data','datasets','models','body_models','checkpoints',
               'outputs','runs','wandb','grab dataset','graspxl dataset'}


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf-8',errors='surrogateescape')


def protected(path, allowed):
    path=path.replace('\\','/'); low=path.lower()
    if path in allowed:
        return False
    first=low.split('/')[0]
    if first in LOCAL_ROOTS or any(first.startswith(p) for p in ('grab_soma','graspxl_soma','graspxl_mano')):
        return True
    if low in {'assets/smplx/version.txt','assets/mano/info.txt','assets/mano/license.txt'}:
        return True
    name=low.rsplit('/',1)[-1]
    return (Path(low).suffix in PAYLOADS or '.zip.part' in name or name.endswith('.tar.gz')
            or (fnmatch.fnmatchcase(name,'.env*') and name!='.env.example'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--index',action='store_true',help='Check the complete proposed commit tree')
    group.add_argument('--history',action='store_true',help='Check HEAD and its ancestors')
    group.add_argument('--all-history',action='store_true',help='Include all local and remote-tracking refs')
    group.add_argument('--pre-push',action='store_true',help='Read Git pre-push ref updates from stdin')
    args=ap.parse_args()
    allowed={line.strip() for line in ALLOWLIST.read_text().splitlines() if line.strip() and not line.startswith('#')}
    if args.history or args.all_history or args.pre_push:
        if git('rev-parse','--is-shallow-repository').strip()=='true':
            print('BLOCKED: shallow history cannot be fully audited.',file=sys.stderr);return 1
        refs=['--all'] if args.all_history else ['HEAD']
        if args.pre_push:
            refs=[]
            for line in sys.stdin:
                fields=line.split()
                if len(fields)!=4 or not re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}',fields[1]):
                    raise ValueError('Invalid pre-push ref update')
                if set(fields[1])!={'0'}:refs.append(fields[1])
            if not refs:return 0
        lines=git('-c','core.quotePath=false','rev-list','--objects',*refs).splitlines()
        paths={line.split(' ',1)[1].strip('"') for line in lines if ' ' in line}
        scope='reachable Git history (including Git LFS pointer paths)'
    else:
        paths=set(git('ls-files','-z').split('\0'))- {''}
        scope='Git index'
    blocked=sorted(p for p in paths if protected(p,allowed))
    if blocked:
        print(f'BLOCKED: {len(blocked)} local model/data paths found in {scope}:',file=sys.stderr)
        for path in blocked:print('  '+path,file=sys.stderr)
        print('Keep files locally. Untrack new files; clean existing history before publishing. See docs/public_release_safety.md.',file=sys.stderr)
        return 1
    print(f'PASS: no disallowed model/data paths in {scope}.')
    return 0


if __name__=='__main__':sys.exit(main())
