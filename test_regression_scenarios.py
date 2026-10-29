import os
import sys
from PIL import Image
from app.prompt_parser import PromptParser
from app.asset_analyzer import AssetAnalyzer
from app.product_integrator import ProductIntegrator
from app.validator import QualityValidator
from app.sitecore_dam import SitecoreDAMClient
from app.engine import OpenSourceDiffusionEngine

print("=" * 60)
print("TEST 1: Jim Jam On Wooden Cafe Table Beside Coffee")
print("=" * 60)
prompt_1 = "Create a realistic advertisement showing this biscuit packet on a wooden café table beside a cup of coffee."
parsed_1 = PromptParser.parse(prompt_1)
print(f"Parsed 1 -> Mode: {parsed_1.interaction_mode}, Env: {parsed_1.scene_environment}, Co-objects: {parsed_1.co_objects}")

dam = SitecoreDAMClient()
ctx_1 = dam.search_brand_and_product(prompt_1, exact_filename="Jim_Jam_BRS_4bbb67699f1.png")
print(f"Asset Retrieved: ID={ctx_1.get('product_id')}, Title={ctx_1.get('title')}")

analyzed_1 = AssetAnalyzer.analyze("reference_cache/ref_primary_69121.png", metadata=ctx_1)
print(f"Asset Analysis -> Has Alpha: {analyzed_1.has_alpha}, Cutout Size: {analyzed_1.cutout_size}, Category: {analyzed_1.object_category}")

flux_pos, flux_neg = PromptParser.build_flux_master_prompt(parsed_1, ctx_1, analyzed_1)
print(f"FLUX.2 Master Prompt:\n{flux_pos}\n")
print(f"FLUX.2 Negative Prompt:\n{flux_neg}\n")

# Verify that negative prompt explicitly forbids insets, rectangles, collages
assert "inset" in flux_neg and "rectangle" in flux_neg and "collage" in flux_neg

print("=" * 60)
print("TEST 2: Jim Jam In A Bright Modern Kitchen")
print("=" * 60)
prompt_2 = "Show this product on a sleek marble counter in a sunlit modern kitchen."
parsed_2 = PromptParser.parse(prompt_2)
print(f"Parsed 2 -> Mode: {parsed_2.interaction_mode}, Env: {parsed_2.scene_environment}")
pos_2, neg_2 = PromptParser.build_flux_master_prompt(parsed_2, ctx_1, analyzed_1)
print(f"FLUX.2 Kitchen Prompt:\n{pos_2[:160]}...\n")

print("=" * 60)
print("TEST 3: Different Product (Britannia Good Day / Pure Magic)")
print("=" * 60)
ctx_3 = dam.search_brand_and_product("Britania GoodDay.jpg", exact_filename="Britania GoodDay.jpg")
print(f"Different Product Asset: ID={ctx_3.get('product_id')}, Title={ctx_3.get('title')}")
prompt_3 = "Place Britania GoodDay.jpg on a rustic wooden picnic table in an outdoor park setting."
parsed_3 = PromptParser.parse(prompt_3)
pos_3, neg_3 = PromptParser.build_flux_master_prompt(parsed_3, ctx_3, None)
print(f"Good Day Prompt:\n{pos_3[:180]}...\n")

print("=" * 60)
print("TEST 4: Quality Validation On Clean Integration")
print("=" * 60)
# Test validating the clean composite
test_img = Image.open("test_flawless_composite.jpg")
report = QualityValidator.validate(test_img, analyzed_1.cutout_image, (1024, 1024))
print(f"Quality Report: Passed={report['passed']}, Score={report['quality_score']}")
print(f"Checks: {report['checks']}")
assert report["passed"] is True, "Quality validation should pass for clean composite"
assert report["checks"]["no_rectangular_borders"] is True, "No rectangular borders should be detected"

print("\nALL 4 REGRESSION TESTS PASSED SUCCESSFULLY!")
