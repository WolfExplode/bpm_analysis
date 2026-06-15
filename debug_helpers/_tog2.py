import sys, tempfile, glob
sys.path.insert(0, '.')
for s in (sys.stdout, sys.stderr):
    try: s.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
import soundfile as sf
from debug_helpers.scan_sequence import _params, _bpm_hint_from_name
from debug_helpers.peak_state_mismatch_detector import find_peak_state_mismatches
from pipeline import analyze_wav_file
OO={k:False for k in ("html","png","csv","summary","debug","filtered_wav","spectrogram","fft_profiles","output_all_passes","working_wav_in_output")}
pats=["#49 *RSA*","#69 *198bpm*","R18心音3 *","玩具熊2*","168_test *","Test4 *","Control [*"]
files=[]
for p in pats:
    g=glob.glob(f"inputs/**/{p}.wav",recursive=True)
    if g: files.append(g[0])
def mm(f,on):
    p=_params(); p["pass3_interval_phase_relabel"]=on
    with tempfile.TemporaryDirectory() as t:
        _,_,_,d=analyze_wav_file(f,p,_bpm_hint_from_name(f),original_file_path=f,output_directory=t,output_options=OO,collect_fft_for_aggregate=False)
    sr=len(d["pass3_state_labels"])/(sf.info(f).frames/sf.info(f).samplerate)
    return len(find_peak_state_mismatches(d.get('peak_classifications') or {},d["pass3_state_boundaries"],sample_rate=sr))
for f in files:
    on=mm(f,True); off=mm(f,False)
    flag = "  <-- OFF better" if off<on else ("  <-- ON better" if on<off else "")
    print(f"  {f.split('/')[-1][:26]:28} ON={on:4} OFF={off:4}{flag}")
