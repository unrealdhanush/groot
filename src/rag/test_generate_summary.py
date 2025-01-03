# test_generate_summary.py

from src.rag.generate_summaries import generate_summary

def main():
    patient_context = "A 65-year-old patient with a history of heart failure and diabetes, predicted high risk of readmission."
    conditions = ["heart failure", "diabetes mellitus"]
    summary = generate_summary(patient_context, conditions)
    print("Generated Summary:")
    print(summary)

if __name__ == "__main__":
    main()
