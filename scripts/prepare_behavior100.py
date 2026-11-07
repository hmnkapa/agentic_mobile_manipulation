"""Freeze one public episode per challenge task, preserving instruction provenance.

The official gallery supplies prose for 50 tasks. The other 50 below are project
translations of the pinned static BDDL goals, not runtime state observations.
No live object IDs, simulator coordinates, scores or solution traces enter them.
The optional mixed-coarse mode retains the 50 official instructions and uses
task-title-level goals for the other 50; their detailed BDDL translations are
not included in the model input. Official scoring is unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter


# Written from task titles, without object counts, BDDL arrangements or a recipe.
COARSE_GOALS = {
    'freeze_fruit': 'Freeze the fruit.',
    'cook_a_brisket': 'Cook a brisket.',
    'sorting_bottles_cans_and_paper': 'Sort the bottles, cans, and paper for recycling.',
    'tidying_living_room': 'Tidy up the living room.',
    'putting_away_toys': 'Put away the toys.',
    're_shelving_library_books': 'Reshelve the library books.',
    'make_rose_centerpieces': 'Make a rose centerpiece.',
    'sweeping_garage': 'Sweep the garage.',
    'stacking_wood': 'Stack the wood.',
    'organizing_art_supplies': 'Organize the art supplies.',
    'scrubbing_bathroom_floor': 'Scrub the bathroom floor.',
    'bringing_paper_to_recycling': 'Take the paper to recycling.',
    'halve_an_egg': 'Halve the egg.',
    'installing_smoke_detectors': 'Install the smoke detector.',
    'setting_the_table': 'Set the table.',
    'unloading_the_car': 'Unload the car.',
    'turning_out_all_lights_before_sleep': 'Turn out all the lights before going to sleep.',
    'boxing_food_after_dinner': 'Pack away the leftover food after dinner.',
    'cleaning_up_branches_and_twigs': 'Clean up the branches and twigs.',
    'vacuuming_floors': 'Vacuum the floor.',
    'thawing_frozen_food': 'Thaw the frozen food.',
    'clean_your_rusty_garden_tools': 'Clean the rusty garden tools.',
    'cook_a_frozen_pie': 'Cook the frozen pie.',
    'organizing_school_stuff': 'Organize the school supplies.',
    'carrying_out_garden_furniture': 'Take the garden furniture outside.',
    'put_together_a_basic_pruning_kit': 'Put together a basic pruning kit.',
    'dispose_of_glass': 'Dispose of the unwanted drinking glasses.',
    'installing_a_modem': 'Set up the modem.',
    'make_cabinet_doors': 'Install the cabinet door.',
    'polishing_shoes': 'Polish the shoes.',
    'clean_up_broken_glass': 'Clean up the broken glass.',
    'packing_meal_for_delivery': 'Pack the meal for delivery.',
    'store_batteries': 'Store the batteries.',
    'store_honey': 'Put away the honey.',
    'tidying_bathroom': 'Tidy up the bathroom.',
    'putting_dirty_dishes_in_sink': 'Put the dirty dishes in the sink.',
    'make_gift_bags_for_baby_showers': 'Prepare gift bags for a baby shower.',
    'collecting_aluminum_cans': 'Collect the aluminum cans.',
    'rearrange_your_room': 'Rearrange the room.',
    'installing_a_fax_machine': 'Set up the fax machine.',
    'composting_waste': 'Compost the food waste.',
    'store_produce': 'Store the produce.',
    'installing_a_scanner': 'Set up the scanner.',
    'clean_a_keyboard': 'Clean the keyboard.',
    'dispose_of_batteries': 'Dispose of the used batteries.',
    'cook_brussels_sprouts': 'Cook the Brussels sprouts.',
    'cook_broccolini': 'Cook the broccolini.',
    'setup_a_bar_for_a_cocktail_party': 'Set up a bar for a cocktail party.',
    'laying_tile_floors': 'Lay the floor tiles.',
    'sorting_books_on_shelf': 'Sort the books on the shelf.',
}


def task_instruction(task, mode='existing'):
    if mode not in {'existing', 'mixed-coarse'}:
        raise ValueError('Unknown instruction mode')
    if task.get('instruction'):
        return task['instruction'], 'official_gallery'
    if mode == 'mixed-coarse':
        return COARSE_GOALS[task['id']], 'project_coarse_task_goal_v1'
    return DERIVED[task['id']], 'project_translation_of_static_bddl_goal'


DERIVED = {
    'freeze_fruit': 'Put the two apples and two strawberries into the two food-storage containers, with one apple and one strawberry in each container. Put both containers inside the kitchen refrigerator and close the refrigerator.',
    'cook_a_brisket': 'Cook the brisket and place it on the chopping board. Leave the frying pan on the kitchen countertop.',
    'sorting_bottles_cans_and_paper': 'Sort the two wine bottles, two soda cans, newspaper and magazine into three separate buckets: both bottles together, both cans together, and the newspaper and magazine together. Keep the three categories in different buckets.',
    'tidying_living_room': 'Put the potted plant and newspaper on the coffee table, the hardback book inside a bookcase, and the notebook on the desk.',
    'putting_away_toys': 'Put all eight toy figures inside the toy boxes. Either of the two toy boxes is acceptable for each figure.',
    're_shelving_library_books': 'Put all three books inside the bookcases; any bookcase is acceptable for each book.',
    'make_rose_centerpieces': 'Put all three roses inside the vase and place the vase on the coffee table.',
    'sweeping_garage': 'Remove both sand and dust from the garage floor.',
    'stacking_wood': 'Arrange the six logs so that every log touches at least one other log.',
    'organizing_art_supplies': 'Put the glue stick, eraser, paintbrush and marker inside the bag, and leave the bag on the desk.',
    'scrubbing_bathroom_floor': 'Remove the dirt from the bathroom floor.',
    'bringing_paper_to_recycling': 'Put the paper sack and newspaper inside the recycling bin, then close the recycling bin.',
    'halve_an_egg': 'Cut the hard-boiled egg into two halves, place both halves on the plate, and put the carving knife inside the sink.',
    'installing_smoke_detectors': 'Attach the smoke detector to the wall nail.',
    'setting_the_table': 'Put both plates on the breakfast table. Put one cupcake on each plate, and place one fork and one table knife next to each plate.',
    'unloading_the_car': 'Bring the briefcase and satchel from the car and place both next to the sofa.',
    'turning_out_all_lights_before_sleep': 'Turn off all three light switches and both table lamps.',
    'boxing_food_after_dinner': 'Put both tacos into the food-storage container, place both plates on top of the sink, put the container inside the kitchen refrigerator, and close the refrigerator.',
    'cleaning_up_branches_and_twigs': 'Put all four branches inside the recycling bin, and leave the recycling bin on the floor.',
    'vacuuming_floors': 'Remove the dust from the floor.',
    'thawing_frozen_food': 'Thaw the chicken breast and bread slice. Leave the chicken breast on a plate inside the microwave, with the microwave closed and turned on. Leave the bread slice on a plate on the countertop. Close the refrigerator.',
    'clean_your_rusty_garden_tools': 'Remove rust from the trowel and scraper, put both tools inside the toolbox, and close the toolbox.',
    'cook_a_frozen_pie': 'Cook the apple pie and leave it on the tray.',
    'organizing_school_stuff': 'Put the folder, book, pencil, pen and calculator inside the bag, and place the bag on the bed.',
    'carrying_out_garden_furniture': 'Move the lawn chair from the living room onto the garden floor, and move the wheelbarrow onto the lawn in the garden.',
    'put_together_a_basic_pruning_kit': 'Put the pruner and shears inside the toolbox, close the toolbox, and leave it on the floor.',
    'dispose_of_glass': 'Put all four drinking glasses inside the trash can.',
    'installing_a_modem': 'Put the modem and television on top of the same living-room cabinet, place the modem next to the television, and turn the modem on.',
    'make_cabinet_doors': 'Attach the cabinet door to the cabinet base.',
    'polishing_shoes': 'Remove dust from both shoes. Put the shoes next to each other and next to the same footstool. Leave the scrub brush on the stand.',
    'clean_up_broken_glass': 'Put all three pieces of broken glass inside the trash can.',
    'packing_meal_for_delivery': 'Put the three wrapped hamburgers into the three paper bags, one hamburger per bag, and place all three bags on top of the storage container.',
    'store_batteries': 'Put all three batteries inside the same cabinet; either cabinet is acceptable.',
    'store_honey': 'Store the jar of honey inside the cabinet in the entryway.',
    'tidying_bathroom': 'Put the bar of soap on the soap dish, the tissue dispenser on the sink, the toilet paper on the toilet, and the cork inside the trash can.',
    'putting_dirty_dishes_in_sink': 'Put both bowls and both plates inside the sink.',
    'make_gift_bags_for_baby_showers': 'Place both gift bags on the coffee table. Put one toy die and one wafer into each bag.',
    'collecting_aluminum_cans': 'Put all six soda cans inside the bucket.',
    'rearrange_your_room': 'Place the tissue dispenser on a stand, and put one of the two pillows on each of the two beds.',
    'installing_a_fax_machine': 'Place the fax machine on a cubicle desk and turn it on.',
    'composting_waste': 'Put the banana half and pomegranate half inside the trash can.',
    'store_produce': 'Put both mangoes and both pomegranates inside the same refrigerator; either refrigerator is acceptable.',
    'installing_a_scanner': 'Place the scanner next to the laptop and turn the scanner on.',
    'clean_a_keyboard': 'Remove the dust from the keyboard.',
    'dispose_of_batteries': 'Put all three batteries inside the trash can, and leave the trash can on the floor.',
    'cook_brussels_sprouts': 'Cut all four Brussels sprouts into halves and cook all eight halves. Leave all halves inside the oven, and close both the oven and refrigerator.',
    'cook_broccolini': 'Cook all three broccolini stalks and both garlic cloves. Leave them on the frying pan, with the frying pan on the stove.',
    'setup_a_bar_for_a_cocktail_party': 'Arrange the wine bottle, all three soda cans, ice bucket, both wineglasses and corkscrew on the countertop. Each soda can should be next to another soda can. Put all six ice cubes inside the bucket. Put the two wineglasses next to each other, with at least one next to the wine bottle. Place the corkscrew next to the wine bottle.',
    'laying_tile_floors': 'Move all four tiles from the corridor onto the bathroom floor. Arrange them so that every tile is next to at least one other tile.',
    'sorting_books_on_shelf': 'Put all three comic books, both notebooks and both hardback books inside the bookcase. Stack the three comic books so that exactly two rest on another comic book. Stack the two notebooks together and the two hardback books together.',
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--official',type=Path,required=True);p.add_argument('--bddl',type=Path,required=True);p.add_argument('--capabilities',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--instruction-mode', choices=['existing','mixed-coarse'], default='existing')
    p.add_argument('--client',choices=['codex','opencode','kimi','qwen'],default='codex')
    p.add_argument('--model')
    p.add_argument('--qwen-config',type=Path)
    a=p.parse_args()
    if (a.client=='qwen')!=(a.qwen_config is not None):p.error('Qwen requires --qwen-config; other clients must omit it')
    raw=json.loads(a.official.read_text());tasks=raw['tasks'] if isinstance(raw,dict) else raw
    assert len(tasks)==len({t['id'] for t in tasks})==100
    caps={t['id']:t for t in json.loads(a.capabilities.read_text())}
    missing={t['id'] for t in tasks if not t.get('instruction')}
    assert missing==set(DERIVED), (missing-set(DERIVED),set(DERIVED)-missing)
    assert missing==set(COARSE_GOALS)
    rows=[]
    for i,t in enumerate(tasks):
        task=t['id'];bddl=(a.bddl/task/'problem0.bddl').read_bytes()
        sha=hashlib.sha256(bddl).hexdigest();assert sha==caps[task]['bddl_sha256']
        instruction,instruction_source=task_instruction(t,a.instruction_mode)
        rows.append({'index':i,'task':task,'name':t['name'],'scene':t['scene_model'],'instance':301,'seed':0,
            'split':'public_test','instruction':instruction,'instruction_sha256':hashlib.sha256(instruction.encode()).hexdigest(),
            'instruction_source':instruction_source,
            'bddl_sha256':sha,'goal_predicates_offline_only':caps[task]['goal_predicates'],
            'candidate_capabilities_offline_only':caps[task]['candidate_capabilities'],
            'run_id':f'mas_b100_{i:03d}_{task.lower()}_i301_s0_r1','status':'planned'})
    result={'schema_version':1,'benchmark':'BEHAVIOR Challenge 2026','protocol':'RGB four fixed cameras + ideal executor; one public instance per task',
        'official_submission_eligible':False,'official_catalog_sha256':hashlib.sha256(a.official.read_bytes()).hexdigest(),
        'instruction_mode':a.instruction_mode,
        'instruction_counts':dict(Counter(row['instruction_source'] for row in rows)),
        'controller_client':a.client+'_cli',
        'model':a.model or ('Qwen/Qwen3.5-4B' if a.client=='qwen' else 'gpt-6.1-sol'),
        'model_reasoning_effort':'low' if a.client=='codex' else None,'agent_profile':'skills',
        'episode_budget_policy':'unbounded','max_agent_steps':None,
        'max_actions':50,'max_sim_steps':20000,'model_timeout_seconds':None,
        'simulator_runtime_max_seconds':None,'startup_timeout_seconds':1200,'tasks':rows}
    if a.qwen_config:result['qwen_config']=str(a.qwen_config.resolve())
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'tasks':len(rows),'instruction_counts':result['instruction_counts']}))


if __name__=='__main__':main()
