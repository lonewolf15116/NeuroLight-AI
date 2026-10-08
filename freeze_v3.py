"""Freeze the v3 protocol. Run ONLY after the reviewer has explicitly approved freezing.
  python freeze_v3.py --i-confirm-freeze
Sets PROTOCOL_V3.json status to FROZEN, then records SHA-256 of every result-determining file
(script, protocol, simulator modules, ARX prior, all model weights) plus the git commit in MANIFEST_V3.json.
confirmation_v3.py refuses to touch confirmation plants unless the files still match this manifest."""
import json, sys, subprocess, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if '--i-confirm-freeze' not in sys.argv: raise SystemExit('Refusing: pass --i-confirm-freeze after explicit reviewer approval.')
p = ROOT / 'PROTOCOL_V3.json'; P = json.loads(p.read_text())
if P['status'] == 'FROZEN': raise SystemExit('Already FROZEN.')
P['status'] = 'FROZEN'; P['frozen_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
p.write_text(json.dumps(P, indent=2))
sys.path.insert(0, str(ROOT)); import confirmation_v3 as C
try: commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
except Exception: commit = 'unavailable'
man = {'frozen_utc': P['frozen_utc'], 'git_commit_before_freeze': commit, 'files': C.manifest_files(P)}
(ROOT / 'MANIFEST_V3.json').write_text(json.dumps(man, indent=2)); print(json.dumps(man, indent=2))
print('FROZEN. Commit PROTOCOL_V3.json and MANIFEST_V3.json before running the confirmation.')
