import os
import numpy as np

input_template = """
tolerance 2.0 
output {output_name}.pdb 
filetype pdb 
structure lj.pdb 
  number {number}
  inside sphere 0. 0. 0. 20.
end structure 
"""

density = 0.5

with open('gen.input', 'w') as f:
    f.write(input_template.format(output_name=f'sphere', number=int(4/3*np.pi*10**3* density)))

os.system('packmol < gen.input')
