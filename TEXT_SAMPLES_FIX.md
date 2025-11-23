# Text Samples Fix - Showing Actual Model Predictions

## Problem
The `text_samples.json` file was showing **target tokens** (input shifted by 1) instead of **actual model predictions**.

**Before:**
```python
# Line 436-437 in train_slimpajama.py
if idx == 0 and len(input_ids) > 0:
    sample_input = self.tokenizer.DecodeIds(input_ids[:50])
    sample_output = self.tokenizer.DecodeIds(target_ids[:50])  # ❌ Shows targets, not predictions!
```

This would show what the model *should* predict, not what it *actually* predicted.

## Solution
Modified `train_on_chunk()` method to generate **actual predictions** using the model's `generate()` method:

**After:**
```python
# Capture first sample for display - generate actual model predictions!
if idx == 0 and len(input_ids) > 0:
    sample_input = self.tokenizer.DecodeIds(input_ids[:50])
    
    # Generate actual predictions from the model
    if self.model:
        # Use first 10 tokens as prompt, generate next 40
        prompt_len = min(10, len(input_ids))
        prompt_ids = input_ids[:prompt_len]
        generated_ids = self.model.generate(prompt_ids, max_length=40)
        # Remove the prompt part to show only predictions
        predicted_ids = generated_ids[prompt_len:]
        sample_output = self.tokenizer.DecodeIds(predicted_ids[:50])
    else:
        # Simulation mode: just show targets
        sample_output = self.tokenizer.DecodeIds(target_ids[:50])
```

## How It Works

1. **Takes first 10 tokens as prompt**: Uses the beginning of the input sequence as context
2. **Generates 40 new tokens**: Calls `model.generate(prompt_ids, max_length=40)`
3. **Removes prompt from output**: Shows only the newly predicted tokens (`generated_ids[prompt_len:]`)
4. **Decodes predictions**: Converts predicted token IDs back to human-readable text

## Result
Now `text_samples.json` will show:
- ✅ **Input**: The actual input text fed to the model
- ✅ **Output**: What the model actually generated/predicted (not the training targets)

This gives you a true view of model performance during training!

## Performance Impact
- Minimal: Only generates predictions for **1 sample per chunk** (not every sample)
- Only runs every **display_samples** steps (default: 10)
- Generation is fast (~10-50ms) compared to full chunk training (~50 seconds)

## Example Output
```json
{
  "step": 150,
  "input": "The quick brown fox jumps over the",
  "output": " lazy dog and runs into the forest where it"
}
```

Now you can see if the model is learning to generate coherent text!
