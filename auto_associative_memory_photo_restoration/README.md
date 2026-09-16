# Photo Restoration Using Auto-associative Memory Network

## Project Title
Photo Restoration Using Auto-associative Memory Network

## Problem Statement
"A restoration tool receives photographs with progressively larger portions missing. Analyze the recalled outputs for each level of missing information, compare the reconstruction behavior, and identify the point at which reliable recall begins to fail."

## Objective
The objective of this project is to build a beginner-friendly Python application that uses a small Hopfield-style auto-associative memory network to restore damaged photographs. The application takes a real photograph uploaded by the user, converts it to a low-resolution grayscale pattern, stores it in memory, applies progressive damage, and analyzes the reconstruction quality.

## Features
- Upload a real photograph
- Convert the photograph to grayscale and low-resolution pattern
- Train a small Hopfield-style auto-associative memory network
- Test damage levels: 10%, 20%, 30%, 40%, 50%, 60%, 70%
- Recall damaged patterns using iterative associative memory
- Compare original, damaged, and recalled images
- Calculate reconstruction accuracy for each damage level
- Plot damage percentage vs. accuracy
- Detect the first level where reliable recall begins to fail
- Download experiment results as CSV
- Save output images and CSV files in a results folder
- Provide reset functionality and beginner-friendly explanations

## Technologies Used
- Python 3
- NumPy
- Pillow
- Matplotlib
- Pandas
- Streamlit

## Installation Steps
1. Open Terminal.
2. Navigate to the project directory.
3. Create a virtual environment if needed:
   python3 -m venv .venv
4. Activate the environment:
   source .venv/bin/activate
5. Install dependencies:
   pip install -r requirements.txt

## How to Run
From the project folder:

streamlit run app.py

## Project Structure
```text
auto_associative_memory_photo_restoration/
├── app.py
├── memory_network.py
├── image_utils.py
├── experiment.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── .gitignore
├── results/
│   └── .gitkeep
├── sample_images/
│   └── README.md
└── __pycache__/
```

## Working Explanation
### What is auto-associative memory?
Auto-associative memory is a system that stores patterns and then recalls them when given a noisy, partial, or incomplete version of the same pattern. It tries to reconstruct the original pattern from the damaged input.

### What is a Hopfield network?
A Hopfield network is a simple recurrent neural network used for associative memory. It stores patterns in a symmetric weight matrix and updates neurons iteratively until the network settles into a stable state.

### What is Hebbian learning?
Hebbian learning says that neurons that fire together become more strongly connected. In a Hopfield network, this is implemented using the outer product of the stored pattern with itself.

### How are weights calculated?
The weight matrix is calculated as:

W_ij = (1/N) sum_{mu} x_i^{mu} x_j^{mu}, for i != j

The diagonal is set to zero to avoid self-connections.

### How is a damaged image recalled?
The damaged image is converted into a bipolar vector. The network repeatedly updates each neuron using the weighted sum of the current state:

s_i = sign(sum_j W_ij s_j)

This iterative update continues until the network reaches a stable state or a fixed number of iterations is completed.

### Why recall can fail when damage increases?
When too many pixels are missing, the input contains too little correct information. The network may converge to the wrong attractor or to a partially wrong memory state. This is why recall reliability decreases as damage increases.

## Results Explanation
The application calculates reconstruction accuracy for each damage level using pixel-by-pixel agreement between the original and recalled bipolar patterns. The results are saved as CSV and displayed in a table and chart. The first damage level where accuracy drops below a chosen reliability threshold is reported as the failure point.

## Limitations
- This is an educational prototype, not a modern high-quality restoration model.
- The Hopfield network is only suitable for small, low-resolution patterns.
- High levels of damage can easily cause incorrect recall.
- Natural photographs are more complex than the compact patterns used in a small Hopfield network.

## References
1. Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. Physical Review Letters.
2. Hertz, J., Krogh, A., and Palmer, R. G. (1991). Introduction to the Theory of Neural Computation.
3. Goodfellow, I., Bengio, Y., and Courville, A. (2016). Deep Learning.

## GitHub Usage Instructions
1. Open your GitHub repository.
2. Add the project folder to the repository.
3. Commit changes using meaningful commit messages.
4. Push the project to GitHub.
5. Share the repository link in your assignment submission.

---

This project is designed to be understandable by a beginner and suitable for viva explanations.
