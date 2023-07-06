import os
import numpy as np

input_template = """
tolerance 2.0 
output {output_name}.pdb 
filetype pdb 
structure lj.pdb 
  number {number}
  inside cube 0. 0. 0. 20. 
end structure 
"""

density = np.linspace(0.1, 1, 1)

for i in density:

    with open('gen.input', 'w') as f:
        f.write(input_template.format(output_name=f'{i:.3f}', number=int(20**3*i)))

    os.system('packmol < gen.input')
