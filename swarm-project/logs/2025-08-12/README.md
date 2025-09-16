## 08-12
  1. Fixing Audio Model Input & Training Pipeline
	•	Discovered that the AudioAgent was throwing shape errors (Dense expecting (batch, 64) but getting (batch, time, 128)).
	•	Adjusted input shape & LSTM handling so the recurrent output was pooled/flattened before passing to Dense layers.
	•	Ensured padded batching produced shape [None, 128] and worked consistently in both training and evaluation.

⸻

2. Verifying Pretraining Works End-to-End
	•	Ran pretraining loop for both agents:
	•	Vision Agent: Frozen backbone (EfficientNetB0 pretrained on ImageNet) + trainable classifier to avoid overfitting.
	•	Audio Agent: Random initialization, learning from scratch.
	•	Confirmed audio agent improved marginally per epoch (slow but stable learning).
	•	Vision agent maintained strong accuracy without rapid overfitting.

⸻

3. Handling Thread Safety & Iterator Issues
	•	Fixed threading barriers to prevent deadlocks when one thread finished early.
	•	Added safe handling for OUT_OF_RANGE exceptions so all threads exit cleanly without hanging.
	•	Resolved earlier issue where the simulation stopped prematurely (~900 rounds) by fixing dataset iterator limits.

⸻

4. Ensuring Audio Data Integrity
	•	Diagnosed that TFRecord audio pipeline was outputting constant -1.0s due to incorrect decoding/normalization.
	•	Compared outputs of TFRecord, MoviePy, and librosa:
	•	Only librosa’s clip was valid initially.
	•	Patched TFRecord reading to normalize audio globally in sim_load_data.py.
	•	Verified TFRecord and librosa now match closely (MAE ≈ 1.8e-5, corr ≈ 1.0).

⸻

5. Full Simulation with Fixed Hedge
	•	Used fixed hedge weights: Vision = 0.95, Audio = 0.05 (to ensure vision dominance given audio’s weakness).
	•	Froze both agents’ weights during fusion to isolate fusion performance.
	•	Logged fusion loss every 100 rounds for clearer trend monitoring.

⸻

6. Results from Full 2.5k-Round Run
	•	Vision agent alone: ~0.83 accuracy final (pretrained classifier only).
	•	Audio agent: low accuracy, as expected, but stable.
	•	Fusion unit: smooth downward loss curve, steady accuracy climb to ~83%.
	•	Achieved stability — no dataset or threading errors.

⸻

7. Next Steps Identified
	•	Gradually reintroduce adaptive hedge:
	•	Use small η (learning rate) for stability.
	•	Add weight clipping & renormalization.
	•	Optionally add temperature scaling to fusion logits for calibration.
	•	Longer term: test with richer datasets & more balanced audio model so fusion can meaningfully outperform vision-only.