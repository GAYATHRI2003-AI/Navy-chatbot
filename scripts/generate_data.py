import json

# Sample Synthetic Data: Instruction, Input, and Technical Response
submarine_data = [
    {
        "instruction": "Troubleshoot a failure in the Forward Trim Tank vent valve.",
        "input": "Depth: 25m. State: Commanded Open, Sensor Indeterminate.",
        "output": "Potential pneumatic actuator failure or mechanical binding. Verify air bank 3 pressure. If pressure is <2000 PSI, switch to backup. If pressure is nominal, manual override via hand-wheel at Frame 42 is required. Monitor pitch bubble to ensure trim remains within +/- 2 degrees."
    },
    {
        "instruction": "Determine the buoyancy compensation needed for a change in water density.",
        "input": "Transition from 35 PSU (Salinity) to 33 PSU. Vessel: SSN-774 class.",
        "output": "Decrease in salinity reduces seawater density. To maintain neutral buoyancy, approximately 8,500 liters of water must be pumped overboard from the Variable Ballast Tanks (VBT). Utilize the auxiliary pump at 50% flow to avoid acoustic transients."
    }
]

# Save to JSONL for Fine-tuning
with open('submarine_qlora_data.jsonl', 'w') as f:
    for entry in submarine_data:
        f.write(json.dumps(entry) + '\n')

print("Dataset generated: submarine_qlora_data.jsonl")