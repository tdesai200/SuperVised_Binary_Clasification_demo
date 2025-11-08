Binary Classification Demo - Instructions
=====================================

Prerequisites
------------
1. Python 3.14 installed on your system
2. Code editor (VS Code recommended)
3. Command line terminal/PowerShell

Step-by-Step Instructions
------------------------

1. Create Virtual Environment
---------------------------
Open PowerShell and run these commands:
python -m venv venv
.\venv\Scripts\activate

2. Install Required Packages
--------------------------
With virtual environment activated, run:
pip install pandas scikit-learn xgboost seaborn matplotlib imbalanced-learn numpy

3. Verify Data Files
------------------
Ensure these files are present in your project directory:
- us_catering_orders.csv (dataset file)
- binary_classification_demo.py (main script)

4. Run the Demo
-------------
Make sure your virtual environment is activated (you'll see (venv) at the start of your command prompt)
Then run:
python binary_classification_demo.py

What to Expect
-------------
The demo will:
1. Load and display dataset information
2. Show class distribution of repeat customers
3. Train a decision tree model
4. Display evaluation metrics including:
   - Confusion Matrix
   - Accuracy
   - Balanced Accuracy
5. Show a visualization of the confusion matrix
6. Test some sample use cases

Important Notes
-------------
- When plots appear, close them to continue program execution
- Keep terminal window open to see all outputs
- Virtual environment must be activated for each new terminal session

Troubleshooting
--------------
If you see errors:
1. Confirm virtual environment is activated (should see (venv) in prompt)
2. Verify all packages are installed: pip list
3. Check if dataset file is in correct location
4. Ensure Python path is correctly set

To Deactivate Virtual Environment
-------------------------------
When finished, type:
deactivate