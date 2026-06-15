import sys, tempfile
sys.path.insert(0, '.')
for s in (sys.stdout, sys.stderr):
    try: s.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass
import soundfile as sf
from debug_helpers.scan_sequence import _params, _bpm_hint_from_name
from debug_helpers.peak_state_mismatch_detector import find_peak_state_mismatches
from debug_helpers.state_sequence_detector import find_sequence_violations
from pipeline import analyze_wav_file
OO={k:False for k in ("html","png","csv","summary","debug","filtered_wav","spectrogram","fft_profiles","output_all_passes","working_wav_in_output")}
f="inputs/Difficulty 3/#49 深呼吸で呼吸性不整脈の心音 Female Respiratory arrhythmias [08-02-2024][irregular][RSA] [107,71-108bpm].wav"
for on in (True, False):
    p=_params(); p["pass3_interval_phase_relabel"]=on
    with tempfile.TemporaryDirectory() as t:
        _,_,_,d=analyze_wav_file(f,p,_bpm_hint_from_name(f),original_file_path=f,output_directory=t,output_options=OO,collect_fft_for_aggregate=False)
    sr=len(d["pass3_state_labels"])/(sf.info(f).frames/sf.info(f).samplerate)
    b=d["pass3_state_boundaries"]
    corr=[c for c in (d.get('pass3_corrections') or []) if c.get('type')=='phase_decision']
    mm=len(find_peak_state_mismatches(d.get('peak_classifications') or {},b,sample_rate=sr))
    print(f"  phase_decision={on}: mismatches={mm} seq={len(find_sequence_violations(b,sample_rate=sr))} phase_corr={corr}")
