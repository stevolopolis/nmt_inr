import gdown


print('Downloading SDF datasets')
gdown.download("https://drive.google.com/uc?id=1xBo6OCGmyWi0qD74EZW4lc45Gs4HXjWw", 'data/stanford3d/gt_armadillo.xyz')
gdown.download("https://drive.google.com/uc?id=1Pm3WHUvJiMJEKUnnhMjB6mUAnR9qhnxm", 'data/stanford3d/gt_dragon.xyz')
gdown.download("https://drive.google.com/uc?id=1wE24AZtXS8jbIIc-amYeEUtlxN8dFYCo", 'data/stanford3d/gt_lucy.xyz')
gdown.download("https://drive.google.com/uc?id=1OVw0JNA-NZtDXVmkf57erqwqDjqmF5Mc", 'data/stanford3d/gt_thai.xyz')