import os
import numpy as np

input_template = """
tolerance 2.0 
output {output_name}.pdb 
filetype pdb 
structure lj.pdb 
  number {number}
  inside ellipsoid 0 0 0 20 10 5 20
end structure 
"""

density = 0.5

with open('gen.input', 'w') as f:
    f.write(input_template.format(output_name=f'cylinder', number=int(4/3*np.pi*20*10*5)))

os.system('packmol < gen.input')
