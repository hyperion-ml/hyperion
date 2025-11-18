# Copyright
#            2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Simple helper script used by the VoxCeleb v1.2 recipe to locate the
# external corpora required during data preparation.  The script is sourced
# by other bash files, so it should only declare environment variables and
# must not execute heavy commands.
#
# Usage:
#   * Edit the path variables below so they point to your local copies of
#     VoxCeleb1, VoxCeleb2, VoxSRC'22 (if available) and MUSAN.
#   * Add a new ``elif`` block if you want to support additional clusters
#     or workstations.  Use the hostname/hostname --domain check that best
#     matches your environment to avoid clobbering other users' settings.
#   * The final ``else`` branch acts as a safety net.  It deliberately exits
#     with an error so that you notice when the dataset paths still need to
#     be configured on a new machine.


if [ "$(hostname -y)" == "clsp" ];then
  # Johns Hopkins/CLSP grid paths.
  voxceleb1_root=/export/corpora5/VoxCeleb1_v2 # VoxCeleb1 v2
  voxceleb2_root=/export/corpora5/VoxCeleb2
  musan_root=/export/corpora5/JHU/musan
elif [ "$(hostname --domain)" == "cm.gemini" ];then
  # Example configuration for the CM Gemini cluster.
  voxceleb1_root=/exp/jvillalba/corpora/voxceleb1 # VoxCeleb1 v2
  voxceleb2_root=/expscratch/dgromero/corpora-open/vox2
  voxsrc22_root=/exp/jvillalba/corpora/voxsrc22
  musan_root=/expscratch/dgromero/corpora-open/musan
else
  echo "[datapath.sh] Please set voxceleb/musan paths for host $(hostname)"
  exit 1
fi

