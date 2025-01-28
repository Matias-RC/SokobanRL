import numpy as np
import pickle

def extract_template_info(template):
    # Extract the 3x3 core from the center
    core_matrix = template[1:4, 1:4]

    # Identify variations from -1 in the border
    variations = []
    center = (2, 2)  # Middle of the matrix is (2, 2) (0-indexed)

    for i in range(5):
        for j in range(5):
            if template[i, j] != -1 and (i < 1 or i > 3 or j < 1 or j > 3):
                # Calculate relative position to the center
                rel_position = (i - center[0], j - center[1])
                variations.append(rel_position)

    return [core_matrix, variations]

def process_templates(templates):
    processed_templates = []
    for template in templates:
        processed_templates.append(extract_template_info(template))
    return processed_templates

template1 = np.matrix([[-1,-1,-1,-1,-1],
                       [-1,0,0,0,-1],
                       [-1,0,0,0,-1],
                       [-1,0,0,0,-1],
                       [-1,-1,-1,-1,-1]])

template2 = np.matrix([[-1,-1,-1,-1,-1],
                       [-1,1,0,0,-1],
                       [-1,0,0,0,-1],
                       [-1,0,0,0,-1],
                       [-1,-1,-1,-1,-1]])

template3 = np.matrix([[-1,-1,-1,0,0],
                       [-1,1,1,0,0],
                       [-1,0,0,0,-1],
                       [-1,0,0,0,-1],
                       [-1,-1,-1,-1,-1]])

template4 = np.matrix([[-1,-1,-1,-1,-1],
                       [-1,1,1,1,-1],
                       [-1,0,0,0,-1],
                       [-1,0,0,0,-1],
                       [-1,-1,-1,-1,-1]])

template5 = np.matrix([[-1,-1,-1,-1,-1],
                       [-1,1,1,1,-1],
                       [-1,1,0,0,-1],
                       [-1,1,0,0,-1],
                       [-1,-1,-1,-1,-1]])

template6 = np.matrix([[-1,-1,0,-1,-1],
                       [-1,1,0,0,-1],
                       [0,0,0,0,-1],
                       [-1,0,0,1,-1],
                       [-1,-1,-1,-1,-1]])

template7 = np.matrix([[-1,-1,-1,-1,-1],
                       [-1,1,0,0,-1],
                       [0,0,0,0,-1],
                       [-1,1,0,0,-1],
                       [-1,-1,-1,-1,-1]])

template8 = np.matrix([[-1,-1,0,-1,-1],
                       [-1,1,0,0,-1],
                       [0,0,0,0,-1],
                       [-1,1,0,1,-1],
                       [-1,-1,0,-1,-1]])

template9 = np.matrix([[-1,-1,0,-1,-1],
                       [-1,1,0,1,-1],
                       [0,0,0,0,0],
                       [-1,1,0,1,-1],
                       [-1,-1,0,-1,-1]])

template10 = np.matrix([[-1,-1,0,-1,-1],
                        [-1,1,0,1,-1],
                        [-1,1,1,1,-1],
                        [-1,1,0,0,0],
                        [-1,-1,-1,-1,-1]])

template11 = np.matrix([[-1,-1,-1,-1,-1],
                       [-1,1,1,1,-1],
                       [0,0,0,0,0],
                       [-1,1,1,1,-1],
                       [-1,-1,-1,-1,-1]])

template12 = np.matrix([[-1,-1,-1,-1,-1],
                       [-1,0,0,0,0],
                       [-1,0,1,0,0],
                       [-1,0,0,0,-1],
                       [-1,-1,-1,-1,-1]])

template13 = np.matrix([[-1,-1,-1,-1,-1],
                       [-1,1,1,1,-1],
                       [-1,1,1,1,-1],
                       [-1,1,1,1,-1],
                       [-1,-1,-1,-1,-1]])

template14 = np.matrix([[-1,-1,-1,-1,-1],
                       [-1,1,1,1,-1],
                       [-1,1,0,0,-1],
                       [0,0,0,0,-1],
                       [0,0,-1,-1,-1]])

template15 = np.matrix([[-1,0,-1,0,-1],
                       [-1,0,0,0,-1],
                       [-1,1,0,1,-1],
                       [-1,0,0,0,-1],
                       [-1,0,-1,0,-1]])

template16 = np.matrix([[-1,-1,-1,-1,-1],
                       [-1,1,1,1,-1],
                       [-1,1,1,1,-1],
                       [-1,0,0,0,-1],
                       [-1,0,0,0,-1]])

template17 = np.matrix([[-1,-1,-1,-1,-1],
                       [-1,1,1,1,-1],
                       [0,0,1,0,0],
                       [-1,0,0,0,-1],
                       [-1,0,0,-1,-1]])

templates = [template1, template2, template3, template4, template5, template6, template7, template8, template9, template10, template11, template12, template13, template14, template15, template16, template17]

processed_templates = process_templates(templates)
with open("templates/templates.pkl", "wb") as f:
    pickle.dump(processed_templates, f)