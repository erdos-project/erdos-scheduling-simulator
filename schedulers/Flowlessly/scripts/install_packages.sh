#!/bin/bash

if [[ ! -f scripts/bash_header.sh ]]; then
  echo "Please run this script from the Flowlessly root directory."
  exit
else
  source scripts/bash_header.sh
fi

OS_ID=$(lsb_release -i -s)
OS_RELEASE=$(lsb_release -r -s)
ARCH_UNAME=$(uname -m)
ARCH=$(get_arch "${ARCH_UNAME}")
ARCHX=$(get_archx "${ARCH_UNAME}")

if [[ -f scripts/pkglist.${OS_ID}-${OS_RELEASE} ]]; then
  source scripts/pkglist.${OS_ID}-${OS_RELEASE}
elif [[ -f scripts/pkglist.${OS_ID}-generic ]]; then
  source scripts/pkglist.${OS_ID}-generic
else
  source scripts/pkglist.generic
fi

UBUNTU_x86_PKGS="${BASE_PKGS} ${COMPILER_PKGS} ${GOOGLE_PKGS} ${BOOST_PKGS}"

USER=$(whoami)
if [[ ${USER} == "root" ]]; then
  NONINTERACTIVE=1
else
  NONINTERACTIVE=${NONINTERACTIVE:-0}
fi
if [[ ${NONINTERACTIVE} -eq 1 ]]; then
  echo "Running as root or with NONINTERACTIVE=1, so will attempt to sort things out non-interactively."
fi

################################################################################
function check_dpkg_packages() {
  print_subhdr "$1 PACKAGE CHECK"
  if [[ $1 == "Ubuntu" ]]; then
    if [[ $2 == "x86" || $2 == "x86_64" ]]; then
      OS_PKGS=${UBUNTU_x86_PKGS}
    else
      echo "No package list available for OS $1 on $2!"
      echo "Sorry, you're on your own now..."
    fi
  fi
  for i in ${OS_PKGS}; do
    PKG_RES=$(dpkg-query -W -f='${Status}\n' ${i} | grep -E "^install" 2>/dev/null)
    if [[ $PKG_RES == "" ]]; then
      MISSING_PKGS="${MISSING_PKGS} ${i}"
    fi
  done

  if [[ $MISSING_PKGS != "" ]]; then
    echo -n "The following packages are required to run ${PROJECT}, "
    echo "but are not currently installed: "
    echo ${MISSING_PKGS}
    if [[ ${NONINTERACTIVE} -eq 1 ]]; then
      sudo apt-get -y install ${MISSING_PKGS}
    else
      echo
      echo "Please install them using the following commmand: "
      echo "$ sudo apt-get install ${MISSING_PKGS}"
      echo
      exit 1
    fi
  else
    echo -n "All required packages are installed."
    echo_success
    echo
  fi
}

################################################################################

print_hdr "INSTALLING NECESSARY PACKAGES"

if [[ ${OS_ID} == "Ubuntu" ]];
then
  echo "Detected ${OS_ID} on ${ARCHX}..."
  echo "Checking if necessary packages are installed..."
  check_dpkg_packages ${OS_ID} ${ARCHX}
else
  echo "Operating systems other than Ubuntu (>=13.10) are not"
  echo "currently supported for automatic configuration."
  ask_continue
fi
