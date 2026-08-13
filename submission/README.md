# Final PDF Generation

The assignment requires one screenshot proving execution on BITS Virtual Lab.

After placing that screenshot in this folder, generate the final PDF:

```powershell
python submission/generate_submission.py --screenshot "submission/BITS_Lab_Execution.png"
```

The output will be `2025AD05095_ML_Assignment_2_Submission.pdf`.

Without `--screenshot`, the generator creates a clearly marked draft PDF that must not be submitted.
