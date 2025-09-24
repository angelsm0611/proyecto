import unittest
import os
from clasificador import predict

class TestPredict(unittest.TestCase):
    def setUp(self):
        self.model_path = "clasificador/models"
        self.sparse_model_name = "logistic_regression"
        self.test_text = """Muscle cramps are a common problem characterized by a sudden, painful, involuntary contraction of muscle. These true cramps, which originate from peripheral nerves, may be distinguished from other muscle pain or spasm. Medical history, physical examination, and a limited laboratory screen help to determine the various causes of muscle cramps. Despite the "benign" nature of cramps, many patients find the symptom very uncomfortable. Treatment options are guided both by experience and by a limited number of therapeutic trials. Quinine sulfate is an effective medication, but the side-effect profile is worrisome, and other membrane-stabilizing drugs are probably just as effective. Patients will benefit from further studies to better define the pathophysiology of muscle cramps and to find more effective medications with fewer side-effects."""

        self.sparse_f1_path = os.path.join(self.model_path, f"{self.sparse_model_name}_val_f1.txt")
        self.biobert_f1_path = os.path.join(self.model_path, "biobert_val_f1.txt")

    def test_predict(self):
        if not os.path.exists(os.path.join(self.model_path, f"{self.sparse_model_name}_model.pkl")):
            self.skipTest(f"Sparse model {self.sparse_model_name} not found")
        if not os.path.exists(os.path.join(self.model_path, "biobert_model")):
            self.skipTest("BioBERT model not found")
        
        label, prob, model_used = predict(
            self.test_text,
            sparse_model_name=self.sparse_model_name,
            model_path=self.model_path,
            sparse_f1_path=self.sparse_f1_path,
            biobert_f1_path=self.biobert_f1_path
        )
        
        self.assertIn(label, [0, 1], "Label should be 0 or 1")
        self.assertTrue(0 <= prob <= 1, "Probability should be between 0 and 1")
        self.assertIn(model_used, ["ensemble", self.sparse_model_name, "biobert"], "Invalid model used")

if __name__ == "__main__":
    unittest.main()