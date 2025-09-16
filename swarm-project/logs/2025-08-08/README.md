## 08-07 (Agent Pretraining Improvements & Error Fixes)
- **Audio agent pretraining**:  
  - Adjusted `AudioAgent` input shape & LSTM handling to prevent `ValueError` from Dense expecting (batch,64) but receiving (batch,time,128).  
  - Fixed embedding extraction and ensured padded batching shape `[None, 128]`.
- **Vision agent unchanged**, still using frozen backbone + trainable classifier.
- **Verified pretraining loop**:  
  - Both agents now run pretraining without shape errors.  
  - Audio agent still improves marginally each epoch (slow learning rate, stable).
- **Threading barrier safety**:  
  - Added logic to avoid deadlocks when one thread finishes early.  
  - Made iterator handling safe for `OUT_OF_RANGE` so all threads exit cleanly.

  ## 08-08 (Full Pretraining + Fusion Unit Run)
- **Full run with ~2.5k rounds**:  
  - Pretrained both agents successfully (Vision ~0.83 acc final, Audio slow but improving).  
  - Ran fusion unit with fixed hedge 0.95 / 0.05.  
  - Achieved smooth loss curve & steady accuracy climb to ~83% final.  
  - No premature stop — confirmed earlier “900 rounds” issue was due to dataset iterator limit in config.
- **Plotted metrics**:  
  - Loss vs. round: clean downward trend.  
  - Accuracy vs. round: steady upward climb, no major instability.
- **Next step planning**:  
  - Re-enable adaptive hedge update with:  
    - Small η for stability.  
    - Clipping & renormalisation.  
    - Optional temperature scaling in fusion logits.  
  - Expect loss curve to get noisier due to dynamic weighting.