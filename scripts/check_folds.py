import json

with open('data/splits/split_manifest.json') as f:
    manifest = json.load(f)

total_rows = 0
for fold in manifest["folds"]:
    total_rows += fold["n_rows"]
    print(f'Fold {fold["fold"]}: quires={fold["quires"]}, n_rows={fold["n_rows"]}')

print(f"\nTotal rows across all folds: {total_rows}")
print("Expected: 4072")
print(f"Match: {total_rows == 4072}")