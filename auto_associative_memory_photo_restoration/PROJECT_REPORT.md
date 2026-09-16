# 1. Title Page
Photo Restoration Using Auto-associative Memory Network

Student: Roshan Antony
Course: Deep Learning Essentials

# 2. Problem Statement
"A restoration tool receives photographs with progressively larger portions missing. Analyze the recalled outputs for each level of missing information, compare the reconstruction behavior, and identify the point at which reliable recall begins to fail."

# 3. Objective
The objective of this project is to implement a beginner-friendly auto-associative memory network that can store a photograph, damage it progressively, and reconstruct it using associative recall. The project also compares reconstruction accuracy across multiple damage levels and identifies the failure point of reliable recall.

# 4. Concept Used
This project uses a Hopfield-style auto-associative memory network. The network stores a compact bipolar representation of the image and uses iterative updates to recall the nearest stored memory from a damaged version of the input.

# 5. Methodology / Working Steps
1. Upload a real photograph.
2. Resize and convert it to grayscale.
3. Convert the image into a bipolar pattern using +1 and -1 values.
4. Train the Hopfield memory network using Hebbian learning.
5. Remove increasingly larger portions of the photograph.
6. Recall the damaged pattern using iterative network updates.
7. Compare the recalled image with the original image.
8. Calculate reconstruction accuracy.
9. Save the output images and CSV results.
10. Analyze the first failure point where reliable recall starts to break down.

# 6. Implementation
## Tools & Libraries
- Python 3
- NumPy
- Pillow
- Matplotlib
- Pandas
- Streamlit

## Source Code / GitHub Repository
The complete source code is available in the GitHub repository:
https://github.com/Roshan2606/auto-associative-memory-photo-restoration

# 7. Results & Output
The application generates the following outputs:
- Original photograph
- Damaged photographs at each tested level
- Recalled photographs at each tested level
- Accuracy table
- Damage vs accuracy chart
- CSV export of results

Placeholder Screenshot 1: Original vs damaged image comparison

Placeholder Screenshot 2: Recalled result after 30% damage

Placeholder Screenshot 3: Recalled result after 60% damage

Placeholder Measured Result Table:
| Damage (%) | Accuracy | Observation |
|-----------|----------|-------------|
| 10 | ____ | ____ |
| 20 | ____ | ____ |
| 30 | ____ | ____ |
| 40 | ____ | ____ |
| 50 | ____ | ____ |
| 60 | ____ | ____ |
| 70 | ____ | ____ |

# 8. Analysis
The Hopfield network works well for small low-resolution patterns and moderate damage. As the amount of missing information increases, the recalled output becomes less stable and may deviate from the original memory state. The network does not perfectly restore natural photographs at high damage levels, which is expected for this educational prototype. The analysis should compare the accuracy values and identify the level at which recall begins to fail reliably.

# 9. Conclusion
This project demonstrates how an auto-associative memory network can be used for a simple educational photograph restoration experiment. The Hopfield model is useful for understanding memory storage, recall, and failure under increasing damage. It is not intended to replace modern image restoration systems, but it is effective for teaching the fundamental concept of associative memory.

# 10. References
1. Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. Physical Review Letters.
2. Hertz, J., Krogh, A., and Palmer, R. G. (1991). Introduction to the Theory of Neural Computation.
3. Goodfellow, I., Bengio, Y., and Courville, A. (2016). Deep Learning.
4. Smith, J. (2024). Auto-associative memory and Hopfield networks in educational demos. (Example reference placeholder)
