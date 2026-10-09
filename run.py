"""Portable entry point. Default verification does not train clinical models."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, urllib.request, platform, importlib.metadata
R=Path(__file__).resolve().parent
def run(path): subprocess.run([sys.executable,str(R/path)],check=True,cwd=R)
def verify():
    manifest=json.loads((R/'SHA256SUMS.json').read_text())
    for name,expected in manifest.items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==expected, 'Changed input: '+name
    sys.path.insert(0,str(R/'core'))
    import audit_core
    audit_core.source_semantics();audit_core.finite()
    run('dca_round_final/analyze.py')
    correlation_audit()
    report={'status':'PASS','scope':'Archived simulation summaries, numerical identities, 9 original and 114 shape first-replicate replays; no clinical refitting or full simulation',
      'integrity_files':len(manifest),'environment':{'python':platform.python_version(),**{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy']}},
      'correlation':json.loads((R/'reruns/correlation_audit/verification.json').read_text()),
      'original':audit_core.REPORT,'shape':json.loads((R/'dca_round_final/analysis/verification.json').read_text())}
    (R/'verification.json').write_text(json.dumps(report,indent=2));print('PASS: verification.json')
def correlation_audit():
    subprocess.run([sys.executable,str(R/'correlation/audit_saved_results.py'),str(R/'correlation/results'),str(R/'reruns/correlation_audit')],check=True,cwd=R)
def fetch():
    for cohort,name in [('support','support2_source.csv'),('gusto','gusto.rda')]:
        p=json.loads((R/'core/protocols'/f'{cohort}.json').read_text())
        url=p.get('url') or p.get('source_url') or p.get('data_url')
        if not url and cohort=='gusto':url='https://raw.githubusercontent.com/resplab/predtools/master/data/gusto.rda'
        expected=p.get('source_sha256') or p.get('data_sha256')
        if not expected and cohort=='gusto':expected='e12bc58730894fa26f31b5b4ea963a878e855d7f2d53e47991cf8bb80b84d8e1'
        out=R/'core/inputs'/cohort/name;out.parent.mkdir(parents=True,exist_ok=True)
        data=out.read_bytes() if out.exists() else urllib.request.urlopen(url,timeout=90).read()
        if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError(f'{cohort}: source hash differs; do not silently use another version')
        tmp=out.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(out);print('Verified source:',cohort)
def clinical():
    for c in ['support','gusto']:run(f'core/analysis/{c}_refit.py')
    # These files remain local and are excluded from the distributable/GitHub tree.
    import zipfile
    target=R/'core/verified_outputs/gusto_for_postprocess.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in (R/'core/verified_outputs/gusto_refit').iterdir():
            if p.is_file():z.write(p,p.name)
        z.write(R/'core/inputs/gusto/gusto.rda','gusto.rda')
    sys.path.insert(0,str(R/'core/analysis'));import postprocess
    postprocess.main(target)
    print('Clinical outputs: core/verified_outputs; compare with reference_clinical. Differences across environments are possible.')
def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['verify','figures','fetch-data','clinical','simulate-original','simulate-shape','verify-correlation','figures-correlation','simulate-correlation','verify-application','verify-binning'])
    a=p.parse_args()
    if a.command=='verify':verify()
    elif a.command=='figures':
        (R/'figures').mkdir(exist_ok=True)
        run('scripts/build_figure1.py');run('scripts/build_figure2.py')
    elif a.command=='verify-binning':run('application/verify_binning.py')
    elif a.command=='verify-application':run('application/analyze_application.py')
    elif a.command=='verify-correlation':correlation_audit()
    elif a.command=='figures-correlation':run('correlation/plot_saved.py')
    elif a.command=='simulate-correlation':
        sys.path.insert(0,str(R/'correlation'));from simulate import run as simulate_correlation
        simulate_correlation(output=R/'reruns/correlation_full')
    elif a.command=='fetch-data':fetch()
    elif a.command=='clinical':clinical()
    elif a.command=='simulate-original':run('core/analysis/finite_generate.py')
    else:
        sys.path.insert(0,str(R/'dca_shape_audit'));from finite_extension import run_finite
        run_finite(R/'reruns/shape',repetitions=500,sizes=(500,3000,10000),seed=2026100807,smoke=False)
if __name__=='__main__':main()
