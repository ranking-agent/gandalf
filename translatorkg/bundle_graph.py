import tarfile

with tarfile.open("gandalf_mmap.tar.gz", "w:gz", format=tarfile.GNU_FORMAT) as t:
    t.add("gandalf_mmap")
