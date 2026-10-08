"""Freeze protocol v4 ONLY after explicit reviewer approval: python freeze_v4.py --i-confirm-freeze"""
import json, sys, subprocess, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if '--i-confirm-freeze' not in sys.argv: raise SystemExit('Refusing: pass --i-confirm-freeze after explicit reviewer approval.')
p = ROOT / 'PROTOCOL_V4.json'; P = json.loads(p.read_text())
if P['status'] == 'FROZEN': raise SystemExit('Already FROZEN.')
P['status'] = 'FROZEN'; P['frozen_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'); p.write_text(json.dumps(P, indent=2))
sys.path.insert(0, str(ROOT)); import confirmation_v4 as C
commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
(ROOT / 'MANIFEST_V4.json').write_text(json.dumps({'frozen_utc': P['frozen_utc'], 'git_commit_before_freeze': commit, 'files': C.manifest_files(P)}, indent=2))
print('FROZEN. Commit PROTOCOL_V4.json and MANIFEST_V4.json before running.')
