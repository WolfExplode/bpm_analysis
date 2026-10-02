## Springer's pipeline was not made to process PCG information at different/changing heart rates. It expects the bpm to be constant across time.
- **Springer:** Calculates **one global HR estimate per recording** before segmentation begins. It uses this single value to parameterize the Gaussian duration distributions (S1, Systole, S2, Diastole) for the **entire sequence**.






S1 -> systole -> S2 -> diastole -> S1...
S1, systole, S2, diastole,
**Training**
ECG → per-frame state labels (S1 / systole / S2 / diastole).
Those labels are used to train logistic regression on the four features (homomorphic envelope, Hilbert envelope, PSD in a band, optional wavelet) and to set duration parameters.

**Segmentation algorithm:**
- Input: PCG only → four features at 50 Hz.
- Observation model: 4-D feature vector → logistic regression → P(state|obs) → Bayes → P(obs|state).
- Structure: 4-state cycle (1→2→3→4→1) with duration distributions from HR and systolic time.
- Decoding: Duration-dependent Viterbi on the whole sequence → one state per frame → that’s the segmentation (and “S1 component from 30 ms to 130 ms” style labels come from this output, not from manual drawing).


**From [springer2015](https://physionet.org/content/hss/1.0/), we have a application of discrete wavelet transform used to create a "profile" of what S1 and S2 sound like:**
The heart sound recordings in the training set were decomposed using the discrete wavelet transform (DWT) with various wavelet families and decomposition levels (These included Haar, Daubechies, symlet, Coiflet, biorthogonal and reverse biorthogonal wavelets at decomposition levels 1-10. Morlet wavelets were not used since a DWT cannot be performed using the Morlet wavelet.)
The absolute value of the detail coefficients for each wavelet family and decomposition level were computed, in order to exclude frequency content outside of the target wavelet range and to extract a positive-valued envelope. The envelope values were summed for each state in the heart sound recordings. The wavelet family and decomposition level that yielded the highest ratio of the sum of the detail coefficients for the S1 and S2 sounds compared to the sum over other intervals across all recordings was selected for further use. This ratio gives an indication of how well each wavelet discriminates between the FHSs and other regions of the heart sound recordings. Therefore, a wavelet envelope computed using such a wavelet would provide the best overall discrimination between the FHSs and other sounds or noise for all heart sound recordings.

They used ECG to get a state for every time sample (S1, S2, Systole, Diastole). To train their segmentation algorithm. 
“training” in this context, means learning the 4 logistic regression models from annotated PCG, so that each frame’s feature vector gets a probability for each of the 4 states.


#### Features:
We build a profile of what each state "looks like". This allows us to get a idea of the characteristics of S1 etc. 

| #   | Feature                    | What it is                                                                                                                                                         | Role                                                                                                                  |
| --- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| 1   | Homomorphic envelope       | Envelope from homomorphic filtering (log → lowpass → exp) on the analytic (Hilbert) signal, then downsampled and normalized.                                       | Main amplitude envelope; S1/S2 show as bumps; follows Schmidt et al.                                                  |
| 2   | Hilbert envelope           | Magnitude of the analytic signal (Hilbert transform), downsampled and normalized.                                                                                  | Another envelope view; used in their training to find S2 center around end-T.                                         |
| 3   | PSD feature                | From a spectrogram: for each time slice, take the mean power in a frequency band (e.g. 40–60 Hz in their code). Resampled to match envelope length and normalized. | Emphasizes frequency content in a band where heart sounds have energy; can help separate S1/S2 from silence or noise. |
| 4   | Wavelet feature (optional) | Detail coefficients at one level of a wavelet decomposition (e.g. level 3, rbio3.9), then downsampled and normalized.                                              | Extra time–frequency detail; can sharpen transitions.                                                                 |
#### How these features are used in the algorithm
**Training**:
Using the ECG-derived per-frame state labels, they collect many feature vectors for each state (1–4). For each state they train a multinomial logistic regression: “given this 4-D feature vector, what is P(state | vector)?” They also compute the overall distribution of the feature vector (mean + covariance over all training data) for use in Bayes’ rule later.
Runtime (**observation model**):
At each frame $t$ they have a 4-D feature vector $ot$​. For each state $j$:
- Logistic regression gives $P(state j∣ot)$.
- They use Bayes to get $P(ot∣state j)$:
- $P(ot∣j)∝P(j∣ot)P(ot)/P(j)$.
- $P(ot)$ is from a multivariate normal fit to all training observations; $P(j)$ is the prior (e.g. from π). So the four features directly determine the emission probabilities for the HMM/HSMM.
**Duration model:**
Independent of the four features: segment lengths (S1, systole, S2, diastole) are modeled with Gaussians (and bounds) derived from heart rate and systolic interval (from autocorrelation on the homomorphic envelope). So:
- Features → “what does this moment look like?” (S1 vs S2 vs systole vs diastole).
- Durations → “how long should each segment be?”
**Viterbi**:
Combines both: at each time and state it maximizes over segment length and previous state, using log P(obs | state) (from the 4-D feature + logistic model) and log P(duration) (from the duration model). Output is the best state sequence over time (segment boundaries and labels 1–4).






















### Viterbi Algorithm
Each frame is given a state. All the frames leading up to this frame form a state sequence. As we progress, we store the best score for each possible state at each frame and use **the model** to assign each state a probability score for this frame. "If I ended up in State X at this frame, what's the BEST way I could have gotten here, and what's that total score?". Then we do the next frame... etc. until we get to the end. Then we end up with a filled table, one column per frame and one row per state. When Viterbi saves scores, it also saves pointers. "To get this Systole score at Frame 4, I came from Systole at Frame 3". At the final frame, we construct the finished state sequence by following the pointers backwards which gives us one state per frame for the whole recording. 

**The model** has three parts:
1. **Transitions**: Which state can follow which (e.g. S1 → systole → S2 → diastole → S1). $\log(a_{ij})$ is either 0 (allowed) or **$-\infty$** (forbidden)
2. **Durations**: A segment of states is a block of consecutive frames that are all assigned the same state. If we modify springer's duration model to use a different BPM per segment, then for every segment, we determine how long each segment tends to last at that bpm. (e.g. this systole segment should last ~23 frames, diastole ~xx frames etc... depending on HR). The duration model is a Hidden Semi-Markov Model (HSMM), it dictates how long you stay in a state before you should to leave. "If I assume the current state segment ends right now, was the duration reasonable?" 
	- The Duration Probability $P(d)$ acts as a gate.
		- If a segment is **too short/long** (unlikely duration), $P(d)$ is very low. This makes the total path score for transitioning to the next state very low, so the algorithm naturally prefers to keep searching for a longer duration in subsequent frames.
		- If a segment is **optimal length** (peak of Gaussian), $P(d)$ is high. This maximizes the path score, making it likely the algorithm will select this path during backtracking.
3. **Emissions**: For every frame, we give one score per state based on how well each state “matches” the observed features in each frame. The emission model we use is logistic regression. 
#### Role of the duration model
In a standard HMM, state duration follows a geometric distribution (implicitly determined by self-transition probability). In an HSMM, we explicitly model how long we stay in a state. Regular Viterbi on a basic HMM would let you flicker S1-Systole-S1-Systole rapidly. But hearts don't do that! The **HSMM enforces physiology**: once you're in S1, you must stay there for a realistic duration. This prevents the algorithm from creating impossible, jittery segmentations.

We modify Springer's duration model to accept time varying bpm.
The original Springer pipeline does not do this: it uses a single global HR (and one systolic time) for the whole recording.

Our modified Springer's approach uses a parametric Gaussian duration distribution conditioned on heart rate:
$$P(d \mid \text{state}, HR) \propto \exp\bigl(-(d - \mu_{\text{state}}(HR))^2 / (2\sigma_{\text{state}}^2)\bigr)$$
Where $\mu_{state}(HR)$ and $\sigma_{state}$ are parametric formulas fitted to [prior research](Segmentation of heart sound recordings by a duration-dependent hidden Markov model, Schmidt et al. (2010)) and time‑varying BPM informs our duration distribution.
At each segment we use the local BPM (and systolic time) to choose the gaussian duration distribution $P(d∣state,HR(t))P(d∣state,HR(t))$ for that segment
S1 and S2 use fixed means; systole and diastole means are functions of HR (and systolic time). Diastole duration varies inversely with heart rate

Why not just hard code some function that takes in heart rate and outputs the expected state durations?
Even at the **same heart rate**, state durations vary significantly between:
- Different patients
- Different cardiac cycles within the same patient
- Different physiological conditions (exercise vs. rest, pathology vs. normal)
A parametric duration distribution will give a [zone of probability](https://www.scribbr.com/wp-content/uploads/2023/02/standard-normal-distribution-example.webp) instead of hard cutoffs.

Instead of tuning parameters, it's much quicker to retrain on pediatric vs. adult vs. pathological populations
*note: When we “re-trained the model” in Python, we only re-fitted the emission model (and total_obs). The duration expectations are still Schmidt’s formulas; they do not adapt to our dataset

#### The HSMM Viterbi Recursion
For each frame $t$ and each state $j$, we consider all possible durations $d$:
$$\delta_t(j) = \max_{i \neq j, d} \left[ \delta_{t-d}(i) + a_{ij} + \log P(d|j, HR_t) + \sum_{\tau=t-d+1}^{t} \log b_j(o_\tau) \right]$$
Where:
- $\delta_t(j)$ = best score to be in state $j$ ending at frame $t$
- $a_{ij}$ = transition log-probability from state $i$ to $j$
- $P(d|j, HR_t)$ = duration probability for staying $d$ frames in state $j$
- $b_j(o_\tau)$ = logistic regression emission probability for observation $o_\tau$ in state $j$

**Key insight:** Instead of frame-by-frame decisions, the HSMM considers "Did I just finish a segment of state $j$ that started $d$ frames ago?"

#### We us logistic regression (LR) for the emission model 
- It gives us greater discrimination between states (S1, Systole, S2, Diastole) based on features (homomorphic, Hilbert, PSD, wavelet). LR-based emissions help the Viterbi choose the right state more often than MVN-based emissions, especially when feature distributions overlap.
- Unlike Gaussian mixtures in traditional HMMs, this is discriminative because the emission model is trained to discriminate between S1 / systole / S2 / diastole (better decision boundaries), rather than generate or model the full distribution of observations in each state. With Gaussians, overlap between states in feature space is modeled as overlapping blobs. With LR you explicitly learn boundaries (or conditional probabilities) that separate the classes, which usually gives better discrimination in overlapping or noisy regions. 
- Outputs are converted to log-probabilities for the Viterbi score
Note: LR training uses random subsampling to balance positive (target state) vs. negative (all other states) samples, because diastole and systole occupy far more frames than S1/S2. Without this, classifiers would be biased toward longer states.
#### Mathematical Summary
**1. Emission Probability (Logistic Regression)**
Springer uses one-vs-all logistic regression. One binary classifier per state, so for each state $j$ you get $P(q_t = j \mid O_t) = \sigma(\beta_j^T x_t)$ from that state’s model. The emission used in the Viterbi is the **likelihood** $b_j(o_t) = P(O_t \mid q_t = j$), obtained from those one-vs-all outputs via Bayes (paper equation 8):
$$b_j(O_t) = \frac{P(q_t = j \mid O_t)\, P(O_t)}{P(q_t = j)}.$$
*(Note: In Viterbi, we use the log of this output directly).*

**2. Duration Probability (Gaussian)**
For state $k$ with expected duration $\mu_k(HR)$ and variance $\sigma_k^2$:
$$ \log p_k(d) = \log \left( \frac{1}{\sqrt{2\pi\sigma_k^2}} \exp \left( -\frac{(d - \mu_k(HR))^2}{2\sigma_k^2} \right) \right) $$
*(Simplified to the quadratic term for optimization often)*.
Also, the four state means must sum to one cardiac period:
$$\mu_{\text{diastole}}(HR(t)) = \frac{60}{HR(t)} - \mu_{S1} - \mu_{\text{systole}}(HR(t)) - \mu_{S2}$$

**3. HSMM Viterbi Recursion (Log-Space)**
$$ \delta_t(j) = \max_{i \in \text{prev}(j)} \max_{d \in [D_{min}, D_{max}]} \left( \delta_{t-d}(i) + \underbrace{\log(a_{ij})}_{0 \text{ or } -\infty} + \log p_j(d) + \sum_{\tau=t-d+1}^{t} \log b_j(o_\tau) \right) $$

**4. Backtracking**
Store $\psi_t(j) = \{i^*, d^*\}$ that maximized the equation above. Reconstruct path from $T$ down to $1$.









## 🔧 Minor Corrections & Clarifications
### 3. **Emission Probability via Bayes' Rule**
Your equation is correct, but the implementation detail matters:
$$b_j(O_t) = \frac{P(q_t = \xi_j | O_t) \cdot P(O_t)}{P(\xi_j)}$$
**Important**: $P(O_t)$ is computed from a **multivariate normal** over the entire training set (all states pooled), not per-state. This acts as a normalization factor. $P(\xi_j) = \pi_j$ is the prior from initial state distribution.
In practice, many implementations (including Springer's) work in **log-odds space** or use the LR outputs directly with appropriate calibration, since the $P(O_t)$ term is state-independent and cancels out during Viterbi decoding when comparing paths.

In `viterbi.py`, `_observation_probs` does:
```python
obs_probs[t, n] = (pihat[t] * Po) / max(pi_vector[n], 1e-10)
```
with `Po = multivariate_normal.pdf(observation_sequence[t], mean=total_obs_mean, cov=total_obs_cov)`. So $P(O_t)$ is one MVN on the **pooled** training observations (your `train.py` builds `total_obs_mean` / `total_obs_cov` from the stacked “all states” matrix). So $b_j(O_t) \propto P(q_t=j\mid O_t)\,P(O_t)/\pi_j$, with $P(O_t)$ the same for all states. In log-space, $\log P(O_t)$ is a constant and doesn’t change which path wins. 

---
### 4. **Viterbi Recursion: Log-Space Nuances**
Your log-space equation is correct, but the **emission term** needs clarification:
$$\sum_{\tau=t-d+1}^{t} \log b_j(o_\tau)$$
This assumes **frame independence** given the state. In reality, consecutive frames within the same state are correlated, but the HSMM assumes conditional independence for tractability.
Also, note that in Springer's extended Viterbi, the indices for the emission product are:
$$\prod_{s=\text{start}_t}^{\text{end}_t} b_j(O_s)$$
Where $\text{start}_t$ and $\text{end}_t$ are clamped to $[1, T]$ to handle boundary conditions.

The recursion uses  
`np.prod(observation_probs[start_t : end_t + 1, j])`  
with `start_t` and `end_t` clamped so the segment stays in `[0, T-1]`. So the emission term is the product over that clamped range, and the model assumes conditional independence of frames given the state. 

---
### 5. **One-Vs-All LR Training**
You mention "one binary classifier per state" — this is correct, but the training requires **balanced subsampling** because the states are highly imbalanced (systole + diastole >> S1 + S2 in duration). Springer explicitly notes this:
> *"Random sub-sampling for each state was performed to ensure that there were a balanced number of samples in each class of the one-versus-all LR training"*
Without this, the classifiers would be biased toward the longer states.

In `train.py`, for each state you build a balanced dataset by subsampling: the “positive” class (that state) and the “negative” class (others) are each limited to a comparable number of samples (e.g. `length_per_other`, `n_select`) so no single state dominates. That matches the paper’s “random sub-sampling for each state… to ensure that there were a balanced number of samples in each class.”

---
### 6. **Extended Viterbi Boundary Handling**
Your description is correct, but the key insight is that the extended algorithm allows the **final state** to end anywhere in $[T, T+d_{max}-1]$, not necessarily at exactly frame $T$. This is crucial for short recordings where the last cardiac cycle might be truncated.
The optimal end time is:
$$T^* = \arg\max_{t \in [T, T+d_{max}-1]} \left[ \max_i \delta_t(i) \right]$$

You have:
```python
temp_delta = delta[T:]
idx_flat = np.argmax(temp_delta)
row_in_temp = idx_flat // N
pos = T + row_in_temp
```
So the best (time, state) is chosen in the extended range starting at `T`, i.e. $T^* \in [T, T + d_{\max}-1]$, and backtracking starts from that `pos`. So the final state can end in that extended window, not necessarily exactly at $T$.









