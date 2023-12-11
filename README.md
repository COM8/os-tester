# OS Tester
A Python pip package to automate testing of whole operating systems with an image recognition based approach and libvirt (qemu). Inspired by openQA.

## Example

![example_when_debug_is_enabled](examples/example.png)

```python
from os_tester.vm import vm
from os_tester.stages import stages
import libvirt

# SELinux Policy for allowing Qemu to access image files:
# ausearch -c 'qemu-system-x86' --raw | audit2allow -M my-qemusystemx86
# semodule -X 300 -i my-qemusystemx86.pp

# NVME: http://blog.frankenmichl.de/2018/02/13/add-nvme-device-to-vm/
# dd if=/dev/zero of=/tmp/test_vm_1.img bs=1M count=8192
# Or:
# qemu-img create -f qcow2 /tmp/test_vm_1.qcow2 8G


def get_vm_xml(name: str, title: str, uuid: str, isoPath: str, vmImagePath: str) -> str:
    ramGiB: int = 2
    numCpus: int = 2

    return f"""
<domain type="kvm" xmlns:qemu='http://libvirt.org/schemas/domain/qemu/1.0'>
  <name>{name}</name>
  <uuid>{uuid}</uuid>
  <title>{title}</title>
  <memory unit="GiB">{ramGiB}</memory>
  <currentMemory unit="GiB">{ramGiB}</currentMemory>
  <vcpu placement="static">{numCpus}</vcpu>
  <os firmware='efi'>
  <!--To get a list of all machines: qemu-system-x86_64 -machine help-->
    <type arch="x86_64" machine="q35">hvm</type>
    <bootmenu enable="yes"/>
    <boot dev="hd"/>
    <boot dev="cdrom"/>
  </os>
  <features>
    <acpi/>
    <apic/>
  </features>
  <clock offset="localtime">
    <timer name="rtc" tickpolicy="catchup"/>
    <timer name="pit" tickpolicy="delay"/>
    <timer name="hpet" present="no"/>
  </clock>
  <on_poweroff>destroy</on_poweroff>
  <on_reboot>restart</on_reboot>
  <on_crash>restart</on_crash>
  <pm>
    <suspend-to-mem enabled="no"/>
    <suspend-to-disk enabled="no"/>
  </pm>
  <devices>
    <emulator>/usr/bin/qemu-system-x86_64</emulator>
    <controller type='pci' index='0' model='pcie-root'/>
    <controller type='pci' index='1' model='pcie-root-port'>
      <model name='pcie-root-port'/>
      <target chassis='1' port='0x10'/>
      <address type='pci' domain='0x0000' bus='0x00' slot='0x02' function='0x0'/>
    </controller>
    <disk type="file" device="cdrom">
      <driver name="qemu" type="raw"/>
      <source file="{isoPath}" startupPolicy="mandatory"/>
      <target dev="hdc" bus="sata"/>
      <readonly/>
      <address type="drive" controller="0" bus="0" target="0" unit="2"/>
    </disk>
    <interface type="user">
      <mac address="52:54:00:8d:ce:97"/>
      <model type="virtio"/>
      <address type="pci" domain="0x0000" bus="0x01" slot="0x00" function="0x00"/>
    </interface>
    <serial type="pty">
      <target type="isa-serial" port="0">
        <model name="isa-serial"/>
      </target>
    </serial>
    <console type="pty">
      <target type="serial" port="0"/>
    </console>
    <input type="tablet" bus="usb">
      <address type="usb" bus="0" port="2"/>
    </input>
    <input type="mouse" bus="ps2"/>
    <input type="keyboard" bus="ps2"/>
    <memballoon model="virtio">
      <address type="pci" domain="0x0000" bus="0x05" slot="0x00" function="0x0"/>
    </memballoon>
  </devices>
  <qemu:commandline>
	  <qemu:arg value='-drive'/>
	  <qemu:arg value='file={vmImagePath},format=qcow2,if=none,id=nvdisk1,media=disk'/>
	  <qemu:arg value='-device'/>
	  <qemu:arg value='nvme,bootindex=1,drive=nvdisk1,serial=1234,id=nvme0,bus=pcie.0,addr=0x04'/>
  </qemu:commandline>
</domain>
    """

if __name__ == "__main__":
    # Connect to qemu
    conn: libvirt.virConnect = libvirt.open("qemu:///system")

    uuid: str = "1e6cae9f-41d7-4fca-8033-fbd538a65173" # Replace with your (random?) UUID
    vmObj: vm = vm(conn, uuid, debugPlt=False)

    # Delete eventually existing VMs
    if vmObj.try_load():
        print(f"Deleting existing VM for UUID '{uuid}'...")
        vmObj.destroy()
        exit(0)
        print(f"VM destroyed.")
    else:
        print(f"No existing VM found for UUID '{uuid}'.")

    # Create and start a new VM
    vmXml: str = get_vm_xml(
        "test_vm_1",
        "Test_VM_1",
        uuid,
        "<PATH_TO_THE_ISO_FILE_TO_BOOT_FROM>",
        "test_vm_1.qcow2", # qemu-img create -f qcow2 test_vm_1.qcow2 8G
    )
    vmObj.create(vmXml)

    # Load stages automation.
    # We expect the `stages.yml` and referenced files inside the stages directory.
    basePath: str = "stages"
    stagesObj: stages = stages(basePath)
    print(stagesObj)
    
    vmObj.run_stages(stagesObj)

    print("All stages done. Exiting...")
    conn.close()
    exit(0)
```

### Stages
Stages are defined as a YAML file. The schema for it is available under [`stages_schema.yml`](stages_schema.yml).
The following shows an example of such a file:
```yaml
stages:
  - stage: Bootloader Selection
    timeout_s: 15
    check:
      file: 0.png
      mse_leq: 0.1
      ssim_geq: 0.99
    actions:
      - keyboard_key:
          value: up
          duration_s: 0.25
      - keyboard_key:
          value: ret
          duration_s: 0.25
  - stage: Installation Started
    timeout_s: 600
    check:
      file: 1.png
      mse_leq: 0.1
      ssim_geq: 0.99
  - stage: Installation Complete
    timeout_s: 600
    check:
      file: 2.png
      mse_leq: 0.1
      ssim_geq: 0.99
    actions:
      - keyboard_key:
          value: tab
          duration_s: 0.25
      - keyboard_key:
          value: tab
          duration_s: 0.25
      - keyboard_key:
          value: ret
          duration_s: 0.25
  - stage: Enter LUKS Password
    timeout_s: 600
    check:
      file: 3.png
      mse_leq: 0.1
      ssim_geq: 0.99
    actions:
      - keyboard_text:
          value: something
          duration_s: 0.25
      - keyboard_key:
          value: ret
          duration_s: 0.25
```

## Building the pip-Package

To build the pip package run:
```bash
python3 -m build
```
The output is then available inside the `dist/` directory.

## pre-commit
Before committing you have to run `pre-commit` to check for linting and type errors.
For this first install `pre-commit`.

```bash
dnf install pre-commit
pre-commit install
```

To run `pre-commit` manually run:
```bash
pre-commit run --all-files
```
