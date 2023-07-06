import os

input_template = """
tolerance 2.0 
output {output_name}.pdb 
filetype pdb 
structure lj.pdb 
  number 100
  inside cube 0. 0. 0. 20. 
end structure 
"""

n_cases = 1

for i in range(n_cases):

    with open('gen.input', 'w') as f:
        f.write(input_template.format(output_name=f'{i}'))

    os.system('packmol < gen.input')
