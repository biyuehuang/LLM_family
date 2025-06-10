# This script will fix the version for most of the key ingredients to make sure the consistent environment no matter when you run it.
# Please run this script with root user
 
if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run as root. Exiting."
  exit 1
fi
 
set -e
WORK_DIR=~/multi-arc
if [ ! -d $WORK_DIR ]; then
        mkdir $WORK_DIR
fi
cd $WORK_DIR
 
export https_proxy=http://child-prc.intel.com:913
export http_proxy=http://child-prc.intel.com:913
export no_proxy=127.0.0.1,*.intel.com
 
echo -e "\n\n############### Install base library ################"
apt install -y vim clinfo build-essential hwinfo net-tools openssh-server curl pkg-config flex bison libelf-dev libssl-dev libncurses-dev git libboost1.83-all-dev cmake libpng-dev docker.io docker-compose-v2
 
echo -e "\n\n############### Add intel repo for PPA and oneapi ###############"
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
add-apt-repository -y ppa:kobuk-team/intel-graphics-testing
apt update
 
echo -e "\n\n############### Install GPU firmware ##############"
cd $WORK_DIR
if [ ! -d firmware ]; then
        mkdir firmware
fi
cd firmware
rm * -rf
wget https://gitlab.com/kernel-firmware/linux-firmware/-/raw/main/xe/bmg_guc_70.bin
wget https://gitlab.com/kernel-firmware/linux-firmware/-/raw/main/xe/bmg_huc.bin
zstd -1 bmg_guc_70.bin -o bmg_guc_70.bin.zst
zstd -1 bmg_huc.bin -o bmg_huc.bin.zst
cp *.zst /lib/firmware/xe
 
echo -e "\n\n############### Install GPU base library ##################"
apt install -y libigdgmm12=22.6.0+ds1-1ubuntu2
apt install -y libigc2=2.5.12-1ubuntu1
 
echo -e "\n\n############### Install Compute related library ##################"
#apt install -y libze-intel-gpu-raytracing=1.1.0-0ubuntu1~25.04~ppa1
#apt install -y libze-intel-gpu1=25.18.33578.11-1~25.04~ppa1
#apt install -y libze1=1.21.9-1~25.04~ppa1
#apt install -y libze-dev=1.21.9-1~25.04~ppa1
#apt install -y intel-opencl-icd=25.18.33578.11-1~25.04~ppa1
apt install -y libze1 libze-dev libze-intel-gpu1 intel-opencl-icd libze-intel-gpu-raytracing
 
echo -e "\n\n############### Install tbb related library ##################"
apt install -y libtbb12=2022.0.0-2
apt install -y libtbbmalloc2=2022.0.0-2
 
echo -e "\n\n############### Install media related library ##################"
apt install -y intel-media-va-driver-non-free=25.1.2+ds1-1ubuntu2
apt install -y vainfo=2.22.0+ds1-2
apt install -y libvpl2=1:2.14.0-1
apt install -y libvpl-tools=1.3.0-1
apt install -y libmfx-gen1.2=25.1.2-1ubuntu2
apt install -y libmfx-gen-dev=25.1.2-1ubuntu2
apt install -y va-driver-all=2.22.0-3ubuntu2
 
echo -e "\n\n############### Install xpu-manager related library ##################"
# libxpum-dev not installed or not found
# libmetee4 not installed or not found
# intel-metrics-discovery not installed or not found
# intel-metrics-library not installed or not found
 
echo -e "\n\n############### Install graphics related library ##################"
apt install -y libegl-mesa0=25.0.3-1ubuntu2
apt install -y libegl1-mesa-dev=25.0.3-1ubuntu2
apt install -y libgl1-mesa-dri=25.0.3-1ubuntu2
apt install -y libgles2-mesa-dev=25.0.3-1ubuntu2
apt install -y libglx-mesa0=25.0.3-1ubuntu2
apt install -y libxatracker2=25.0.3-1ubuntu2
apt install -y mesa-va-drivers=25.0.3-1ubuntu2
apt install -y mesa-vdpau-drivers=25.0.3-1ubuntu2
apt install -y mesa-vulkan-drivers=25.0.3-1ubuntu2
 
echo -e "\n\n############### Install oneapi ##################"
apt install -y intel-oneapi-base-toolkit=2025.1.3-6
 
echo -e "\n\n############### Install level-zero-tests for P2P benchmark ##################"
cd $WORK_DIR
rm level-zero-tests -rf
git clone https://github.com/oneapi-src/level-zero-tests.git
cd level-zero-tests
git checkout 6f4258713c57ed1668671e5c016633624602184d
mkdir build
cd build
cmake ../
make -j$(nproc)
cd $WORK_DIR
 
echo -e "\n\n################ Generate script to set performance mode ####################"
cd $WORK_DIR
TARGET_SCRIPT="setup_perf.sh"
cat << 'EOF' > "$TARGET_SCRIPT"
#!/bin/bash
gpu_num=`sudo xpu-smi discovery | grep card | wc -l`
for((i=0; i<$gpu_num; i++)); do
  echo "Set GPU $i freq to 2400Mhz"
  sudo xpu-smi config -d $i -t 0 --frequencyrange 2400,2400
done
 
echo "Set CPU to performance mode"
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor && echo 0 | sudo tee /sys/devices/system/cpu/cpu*/power/energy_perf_bias
EOF
chmod +x "$TARGET_SCRIPT"
 
echo -e "\n\n################ System Configuration #####################"
gpasswd -a ${USER} render
newgrp render
sed -i "s/.*WaylandEnable=false/WaylandEnable=true/" /etc/gdm3/custom.conf
update-initramfs -u
 
echo -e "\n\n################ Disable intel_iommu ######################"
GRUB_FILE="/etc/default/grub"
cp "$GRUB_FILE" "${GRUB_FILE}.bak"
sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=off"/' "$GRUB_FILE"
update-grub
