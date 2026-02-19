import json

indian_navy_dataset = [
    {
        "instruction": "Verify if a fuel transfer maneuver is permitted in the Lakshadweep marine protected zone under current Indian Navy environmental guidelines.",
        "input": "Zone: Lakshadweep Marine National Park. Activity: Ship-to-ship fuel transfer. Current Sea State: 3.",
        "output": "Negative. Under Indian Navy Environmental Protection Policy (INEPP) and MARPOL Annex I, ship-to-ship fuel transfers are strictly prohibited within 12 nautical miles of the Lakshadweep Marine National Park to prevent accidental coral reef contamination. Maneuver must be delayed until the vessel is in open waters (Deep Sea) or conducted at a designated naval port facility."
    },
    {
        "instruction": "Provide the immediate action checklist for a 'Fire in the Galley' on a Talwar-class frigate.",
        "input": "Scenario: Class B fire (Oil/Grease) in the primary galley. Ship state: Action Stations.",
        "output": "1. Sound the local fire alarm and report to Damage Control Headquarters (DCHQ). 2. Isolate electrical power and fuel supply to the galley immediately. 3. Deploy Aqueous Film Forming Foam (AFFF) or CO2 extinguishers; do NOT use water on grease fires. 4. Activate the galley's fixed fire suppression system (Halon/FM-200 substitute). 5. Seal all fire-tight doors and ventilation dampers to the galley compartment."
    },
    {
        "instruction": "Explain the 'Emergency Deep' protocol for an Arihant-class submarine.",
        "input": "Threat: Incoming airborne ASW (Anti-Submarine Warfare) torpedo detected.",
        "output": "1. Order 'Emergency Deep' on the ship's main broadcast. 2. Set diving planes to hard dive (30-35 degrees). 3. Increase propulsion to maximum revolutions (Flank Speed). 4. Flood all internal compensating tanks to add negative buoyancy. 5. Level off below the primary thermocline layer to exploit acoustic shadow zones. 6. Deploy acoustic decoys (Nayan/Maareech systems) if available."
    },
    {
        "instruction": "What are the rules for 'Gray Water' discharge in the Gulf of Mannar?",
        "input": "Vessel type: Scorpene-class. Position: 5nm off the coast.",
        "output": "Prohibited. The Gulf of Mannar is a Biosphere Reserve. Indian Navy Environmental SOPs mandate Zero-Discharge within 12nm of coastal marine reserves. All gray water must be stored in holding tanks until the vessel reaches open sea (beyond 24nm) or is processed by shipboard treatment plants to IMV standards."
    },
    {
        "instruction": "Emergency protocol for 'Hull Breach' in Forward Torpedo Room.",
        "input": "Depth: 100m. Severity: Rapid flooding.",
        "output": "1. IMMEDIATE: Close water-tight doors to the Torpedo Room. 2. Blow Emergency Main Ballast (EMB) to initiate rapid ascent. 3. Personnel evacuation to mid-ship section. 4. Isolate Torpedo tube hydraulics. 5. Notify Command of depth excursion limits."
    }
]

with open('c:/Users/Yazhini/OneDrive/Desktop/aero/data/indian_navy_qlora.jsonl', 'w') as f:
    for entry in indian_navy_dataset:
        f.write(json.dumps(entry) + '\n')
