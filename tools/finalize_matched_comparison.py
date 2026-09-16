"""Add cross-dataset counts, reviewed previews and selected optional assets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--extras', type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    manifest = json.loads((package / 'manifest.json').read_text())
    validation = json.loads((package / 'reports/validation.json').read_text())
    assert manifest['status'] == 'converted_and_numerically_validated' and validation['passed']
    grab_root = package.parent / 'GRAB_SOMA_51Objects'
    grab_validation = json.loads((grab_root / 'reports/validation.json').read_text())
    assert grab_validation['status'] == 'validated'
    inventory = json.loads((ROOT / 'out/grab-preparation/inventory.json').read_text())
    grab_counts = Counter(r['object'] for r in inventory['sequences'])
    grab_frames = Counter()
    for r in inventory['sequences']: grab_frames[r['object']] += r['frames']
    frames = Counter()
    for r in json.loads((package / 'metadata/sequence_index.json').read_text()): frames[r['object_key']] += r['frames']
    rows = []
    lines = ['# Motion counts by matched object', '',
             'GRAB frame counts use 120 FPS. GraspXL source FPS is unspecified; its frame counts do not imply an equivalent duration. '
             'These are all available selected-size diverse MANO motions. Tabletop raw source files, if supplied, are separate and excluded.', '',
             '| GRAB object | Match | GRAB sequences | GRAB frames | GraspXL sequences | GraspXL frames |',
             '|---|---|---:|---:|---:|---:|']
    for obj in manifest['objects']:
        name = obj['grab_object']
        row = dict(grab_object=name, match_type=obj['match_type'], same_type_evaluation_eligible=obj['match_type']=='same_type',
                   grasp_source_uid=obj['uid'], grasp_size_variant=obj['size_variant'],
                   grab_sequences=grab_counts[name], grab_frames=grab_frames[name],
                   graspxl_sequences=obj['sequence_count'], graspxl_frames=frames[obj['object_key']],
                   grab_extents_cm=obj['grab_extents_cm'],graspxl_extents_cm=obj['axis_aligned_extents_cm'])
        rows.append(row)
        lines.append(f'| {name} | {row["match_type"]} | {row["grab_sequences"]} | {row["grab_frames"]} | {row["graspxl_sequences"]} | {row["graspxl_frames"]} |')
    (package / 'MOTION_COUNTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (package / 'metadata/dataset_comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    preview = ROOT / 'out/grab-matched-graspxl/converted-mug-preview.png'
    if preview.exists(): shutil.copyfile(preview, package / 'previews/converted-mug-preview.png')
    if args.extras:
        assert args.extras.is_dir()
        destination = package / 'extra_assets'
        if destination.exists(): raise FileExistsError(destination)
        shutil.copytree(args.extras, destination)
    readme = package / 'README.md'
    base = readme.read_text(encoding='utf-8').replace('--object-number 1 --sequence-index 0','--object-number 29 --sequence-index 0')
    suffix = '\n## Dataset comparison and additional files\n\nSee [COMPARISON.md](COMPARISON.md) for object-type and size differences, [MOTION_COUNTS.md](MOTION_COUNTS.md) for every object\'s counts, and `metadata/dataset_comparison.json` for a machine-readable table. The converted mug preview shows three saved SOMA frames.\n'
    if args.extras:
        suffix += '\n`extra_assets/` contains selected original simulator descriptions and/or tabletop source motions with extraction reports. These are separate source assets: tabletop motions are not converted SOMA motions, and source URDFs still require Isaac import and physics validation. Read the extraction report before choosing an asset variant.\n'
        start = base.index('## Downloads for the next stage')
        stop = base.index('## Sources and attribution', start)
        base = base[:start] + '## Downloaded extras\n\nThe matching source simulator assets and available tabletop motions have been extracted under `extra_assets/`. Read its report and README. Tabletop motions use a different MANO convention and are retained as raw source data for a separate conversion.\n\n' + base[stop:]
    readme.write_text(base+suffix,encoding='utf-8')
    hashes = {p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in package.rglob('*') if p.is_file() and p.name != 'checksums.json'}
    (package / 'checksums.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    result = dict(objects=manifest['object_count'],unique_source_ids=manifest['unique_source_object_ids'],
                  sequences=manifest['sequence_count'],frames=manifest['frame_count'],
                  match_types=manifest['match_types'], package_bytes=sum(p.stat().st_size for p in package.rglob('*') if p.is_file()))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
