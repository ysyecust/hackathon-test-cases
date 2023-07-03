import os
import numpy as np

input_template = """
tolerance 2.0 
output {output_name}.pdb 
filetype pdb 
structure lj.pdb 
  number {number}
  inside cylinder 0. 0. 0. 1. 0. 0. 10. 20.
end structure 
"""

density = 0.3

with open('gen.input', 'w') as f:
    f.write(input_template.format(output_name=f'cylinder', number=int(np.pi*10**2*20* density)))

os.system('packmol < gen.input')
