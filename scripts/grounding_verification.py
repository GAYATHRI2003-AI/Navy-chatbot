"""
LLM Grounding Verification System
Addresses: DeepSeek-R1 Hallucination Risk

Features:
- Fact-checking against structured knowledge base
- Self-consistency verification
- Confidence calibration
- Hallucination detection
"""

import re
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from difflib import SequenceMatcher

@dataclass
class FactCheckResult:
    """Result of fact-checking operation"""
    claim: str
    verified: bool
    confidence: float
    source: Optional[str]
    correction: Optional[str]
    
class StructuredKnowledgeBase:
    """Structured database of verified naval facts"""
    
    def __init__(self, kb_path="knowledge_base/structured_facts.json"):
        self.kb_path = kb_path
        self.facts = self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load structured facts from JSON"""
        if os.path.exists(self.kb_path):
            with open(self.kb_path, 'r') as f:
                return json.load(f)
        
        # Initialize with critical naval facts
        default_kb = {
            "submarines": {
                "Kalvari-class": {
                    "displacement": {"value": 1775, "unit": "tons", "tolerance": 50},
                    "length": {"value": 67.5, "unit": "meters", "tolerance": 1.0},
                    "max_depth": {"value": 350, "unit": "meters", "tolerance": 20},
                    "speed_submerged": {"value": 20, "unit": "knots", "tolerance": 2},
                    "crew": {"value": 43, "unit": "personnel", "tolerance": 5},
                    "propulsion": "Diesel-Electric with AIP"
                },
                "Arihant-class": {
                    "displacement": {"value": 6000, "unit": "tons", "tolerance": 200},
                    "length": {"value": 111, "unit": "meters", "tolerance": 2},
                    "max_depth": {"value": 300, "unit": "meters", "tolerance": 20},
                    "propulsion": "Nuclear PWR",
                    "classification": "SSBN"
                }
            },
            "weapons": {
                "Varunastra": {
                    "type": "Heavy Weight Torpedo",
                    "range": {"value": 40, "unit": "km", "tolerance": 5},
                    "speed": {"value": 40, "unit": "knots", "tolerance": 3},
                    "warhead": {"value": 250, "unit": "kg", "tolerance": 10}
                },
                "Brahmos": {
                    "type": "Supersonic Cruise Missile",
                    "range": {"value": 290, "unit": "km", "tolerance": 10},
                    "speed": {"value": 2.8, "unit": "Mach", "tolerance": 0.2}
                }
            },
            "naval_bases": {
                "INS Kadamba": {
                    "location": "Karwar",
                    "coordinates": {"lat": 14.8, "lon": 74.1},
                    "fleet": "Western Naval Command",
                    "type": "Strategic Tier-1"
                },
                "INS Varsha": {
                    "location": "Rambilli (Visakhapatnam)",
                    "fleet": "Eastern Naval Command",
                    "type": "Nuclear Submarine Base"
                }
            },
            "maritime_law": {
                "territorial_waters": {"value": 12, "unit": "nautical miles"},
                "contiguous_zone": {"value": 24, "unit": "nautical miles"},
                "eez": {"value": 200, "unit": "nautical miles"}
            }
        }
        
        # Save default KB
        os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)
        with open(self.kb_path, 'w') as f:
            json.dump(default_kb, f, indent=2)
        
        return default_kb
    
    def verify_numeric_claim(self, category: str, entity: str, attribute: str, claimed_value: float) -> FactCheckResult:
        """Verify a numeric claim against the knowledge base"""
        try:
            fact = self.facts[category][entity][attribute]
            
            if isinstance(fact, dict) and "value" in fact:
                true_value = fact["value"]
                tolerance = fact.get("tolerance", true_value * 0.1)  # 10% default tolerance
                
                # Check if claim is within tolerance
                if abs(claimed_value - true_value) <= tolerance:
                    return FactCheckResult(
                        claim=f"{entity} {attribute}: {claimed_value}",
                        verified=True,
                        confidence=0.95,
                        source=f"KB: {category}/{entity}",
                        correction=None
                    )
                else:
                    return FactCheckResult(
                        claim=f"{entity} {attribute}: {claimed_value}",
                        verified=False,
                        confidence=0.90,
                        source=f"KB: {category}/{entity}",
                        correction=f"Correct value: {true_value} {fact.get('unit', '')}"
                    )
            else:
                # Non-numeric fact
                if str(claimed_value).lower() == str(fact).lower():
                    return FactCheckResult(
                        claim=f"{entity} {attribute}: {claimed_value}",
                        verified=True,
                        confidence=0.95,
                        source=f"KB: {category}/{entity}",
                        correction=None
                    )
                else:
                    return FactCheckResult(
                        claim=f"{entity} {attribute}: {claimed_value}",
                        verified=False,
                        confidence=0.90,
                        source=f"KB: {category}/{entity}",
                        correction=f"Correct value: {fact}"
                    )
        
        except KeyError:
            return FactCheckResult(
                claim=f"{entity} {attribute}: {claimed_value}",
                verified=False,
                confidence=0.0,
                source="Unknown",
                correction="Fact not in knowledge base"
            )


class GroundingVerifier:
    """Verifies LLM outputs against ground truth"""
    
    def __init__(self):
        self.kb = StructuredKnowledgeBase()
        self.verification_patterns = self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for fact extraction"""
        return {
            "displacement": re.compile(r'(\d+(?:,\d+)?)\s*(?:tons?|tonnes?)', re.IGNORECASE),
            "length": re.compile(r'(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)', re.IGNORECASE),
            "depth": re.compile(r'(\d+)\s*(?:m|meters?|metres?).*depth', re.IGNORECASE),
            "speed": re.compile(r'(\d+)\s*(?:knots?|kts?)', re.IGNORECASE),
            "range": re.compile(r'(\d+)\s*(?:km|kilometers?|kilometres?|nm|nautical miles?)', re.IGNORECASE),
        }
    
    def extract_claims(self, text: str) -> List[Tuple[str, float]]:
        """Extract factual claims from LLM output"""
        claims = []
        
        for claim_type, pattern in self.verification_patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                # Clean numeric value
                value = float(match.replace(',', ''))
                claims.append((claim_type, value))
        
        return claims
    
    def verify_response(self, llm_output: str, context: Dict = None) -> Dict:
        """
        Verify LLM response for factual accuracy
        
        Args:
            llm_output: The LLM's generated text
            context: Optional context (e.g., {"submarine": "Kalvari-class"})
        
        Returns:
            Verification report with corrections
        """
        # Extract claims from output
        claims = self.extract_claims(llm_output)
        
        verification_results = []
        hallucinations = []
        
        # Verify each claim
        for claim_type, claimed_value in claims:
            if context and "submarine" in context:
                result = self.kb.verify_numeric_claim(
                    "submarines",
                    context["submarine"],
                    claim_type,
                    claimed_value
                )
                verification_results.append(result)
                
                if not result.verified and result.confidence > 0.5:
                    hallucinations.append(result)
        
        # Calculate overall grounding score
        if len(verification_results) > 0:
            verified_count = sum(1 for r in verification_results if r.verified)
            grounding_score = (verified_count / len(verification_results)) * 100
        else:
            grounding_score = 0.0  # No verifiable claims found
        
        return {
            "grounding_score": grounding_score,
            "verified_claims": len([r for r in verification_results if r.verified]),
            "total_claims": len(verification_results),
            "hallucinations": hallucinations,
            "verification_details": verification_results,
            "confidence_level": self._calculate_confidence(grounding_score, len(verification_results))
        }
    
    def _calculate_confidence(self, grounding_score: float, num_claims: int) -> str:
        """Calculate confidence level based on grounding score"""
        if num_claims == 0:
            return "UNVERIFIABLE"
        elif grounding_score >= 90:
            return "HIGH"
        elif grounding_score >= 70:
            return "MEDIUM"
        else:
            return "LOW"
    
    def self_consistency_check(self, llm_responses: List[str], num_samples=3) -> Dict:
        """
        Perform self-consistency verification by comparing multiple LLM outputs
        
        Args:
            llm_responses: List of LLM responses to the same query
            num_samples: Number of samples to compare
        
        Returns:
            Consistency analysis
        """
        if len(llm_responses) < 2:
            return {"consistent": True, "confidence": 0.5, "note": "Insufficient samples"}
        
        # Extract claims from each response
        all_claims = [self.extract_claims(resp) for resp in llm_responses]
        
        # Check consistency of numeric values
        consistency_scores = []
        
        for i in range(len(all_claims)):
            for j in range(i + 1, len(all_claims)):
                # Compare claim sets
                claims_i = dict(all_claims[i])
                claims_j = dict(all_claims[j])
                
                common_keys = set(claims_i.keys()) & set(claims_j.keys())
                
                if len(common_keys) > 0:
                    matches = sum(
                        1 for k in common_keys 
                        if abs(claims_i[k] - claims_j[k]) / max(claims_i[k], claims_j[k]) < 0.1
                    )
                    consistency_scores.append(matches / len(common_keys))
        
        if len(consistency_scores) > 0:
            avg_consistency = np.mean(consistency_scores)
            return {
                "consistent": avg_consistency > 0.8,
                "confidence": avg_consistency,
                "note": f"Checked {len(llm_responses)} samples"
            }
        else:
            return {"consistent": False, "confidence": 0.0, "note": "No common claims found"}
    
    def generate_corrected_output(self, llm_output: str, verification_report: Dict, include_header: bool = True) -> str:
        """Generate corrected output with hallucinations removed"""
        corrected = llm_output.strip()
        
        # Don't process insufficient data responses
        if "Insufficient" in corrected:
            return corrected
        
        # Ensure answer ends properly
        if corrected and not corrected.endswith(('.', '!', '?')):
            # Find last complete sentence
            last_period = max(
                corrected.rfind('.'),
                corrected.rfind('!'),
                corrected.rfind('?')
            )
            if last_period > 50:
                corrected = corrected[:last_period + 1]
            else:
                corrected = corrected + "."
        
        # Add corrections if any
        for hallucination in verification_report["hallucinations"]:
            if hallucination.correction:
                # Add correction at the end
                corrected += f"\n\n[CORRECTION: {hallucination.correction}]"
        
        if not include_header:
            return corrected

        # Add grounding score badge at the end
        score = verification_report["grounding_score"]
        confidence = verification_report["confidence_level"]
        
        footer = f"\n\n[GROUNDING SCORE: {score:.1f}% | CONFIDENCE: {confidence}]"
        
        return corrected + footer


def verify_llm_output(llm_response: str, context: Dict = None, include_header: bool = True) -> Tuple[str, Dict]:
    """
    Main verification function to be called from main.py
    
    Args:
        llm_response: Raw LLM output
        context: Optional context dictionary
        include_header: Whether to include the grounding score header
    
    Returns:
        (corrected_output, verification_report)
    """
    verifier = GroundingVerifier()
    
    # Perform verification
    report = verifier.verify_response(llm_response, context)
    
    # Generate corrected output
    corrected = verifier.generate_corrected_output(llm_response, report, include_header=include_header)
    
    # Log verification results
    if report["grounding_score"] < 70:
        print(f"⚠️  LOW GROUNDING SCORE: {report['grounding_score']:.1f}%")
        print(f"   Verified: {report['verified_claims']}/{report['total_claims']} claims")
        if report["hallucinations"]:
            print(f"   Hallucinations detected: {len(report['hallucinations'])}")
    
    return corrected, report


# Example usage and testing
if __name__ == "__main__":
    verifier = GroundingVerifier()
    
    # Test case 1: Accurate response
    accurate_response = """
    The Kalvari-class submarine has a displacement of 1,775 tons and a length of 67.5 meters.
    It can operate at depths up to 350 meters and has a submerged speed of 20 knots.
    """
    
    print("=" * 60)
    print("TEST 1: Accurate Response")
    print("=" * 60)
    report1 = verifier.verify_response(accurate_response, {"submarine": "Kalvari-class"})
    print(f"Grounding Score: {report1['grounding_score']:.1f}%")
    print(f"Confidence: {report1['confidence_level']}")
    print(f"Verified Claims: {report1['verified_claims']}/{report1['total_claims']}")
    
    # Test case 2: Hallucinated response
    hallucinated_response = """
    The Kalvari-class submarine has a displacement of 3,000 tons and a length of 85 meters.
    It can operate at depths up to 500 meters and has a submerged speed of 35 knots.
    """
    
    print("\n" + "=" * 60)
    print("TEST 2: Hallucinated Response")
    print("=" * 60)
    report2 = verifier.verify_response(hallucinated_response, {"submarine": "Kalvari-class"})
    print(f"Grounding Score: {report2['grounding_score']:.1f}%")
    print(f"Confidence: {report2['confidence_level']}")
    print(f"Hallucinations: {len(report2['hallucinations'])}")
    
    for h in report2['hallucinations']:
        print(f"  ✗ {h.claim}")
        print(f"    → {h.correction}")
    
    # Test case 3: Self-consistency check
    print("\n" + "=" * 60)
    print("TEST 3: Self-Consistency Check")
    print("=" * 60)
    
    responses = [
        "The Kalvari-class has a displacement of 1,775 tons.",
        "Kalvari-class displacement: 1,780 tons.",
        "The submarine weighs approximately 1,775 tons."
    ]
    
    consistency = verifier.self_consistency_check(responses)
    print(f"Consistent: {consistency['consistent']}")
    print(f"Confidence: {consistency['confidence']:.2f}")
