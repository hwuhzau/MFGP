import subprocess

step = ["s01", "s02", "s03", "s12", "s13", "s23", "s012", "s013", "s023", "s123", "s0123"]

for stepi in step:
    for seedi in range(1, 11):
        subprocess.run(
            ['python', 'train.py', '--step', f"{stepi}", '--seed', f"{seedi}"])