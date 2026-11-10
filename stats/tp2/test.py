import io_utils
from cpm_reader import compare
p_ref = io_utils.load_snapshot("../../snaps_vizu/snapshot_it_000250.csv")
print(compare("../../snaps_int16/compressed/snapshot_it_000250.cpm", p_ref))