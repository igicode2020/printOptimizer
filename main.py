import json
from pyomo.environ import *
from datetime import timedelta
import random

with open('printers.json', 'r') as f:
    printers = json.load(f)

with open('models.json', 'r') as f:
    models = json.load(f)

with open('task.txt', 'r') as f:
    task_list = [line.strip() for line in f if line.strip()]

def parse_time(timestr):
    h, m, s = map(int, timestr.split(':'))
    return h + m/60 + s/3600

def predict_print_time_ml(model_info):
    base = len(model_info.get('model_name', ''))
    return max(0.5, base * 0.2 + random.uniform(-0.2, 0.2))

jobs = []
for idx, model_id in enumerate(task_list):
    model_info = next((m for m in models if m['file_id'] == model_id), None)
    if model_info:
        print_time = predict_print_time_ml(model_info)
        jobs.append({'job_id': idx, 'model_id': model_id, 'model': model_info, 'print_time': print_time})
    else:
        raise ValueError(f"Model ID {model_id} not found in models.json")

printer_ids = [p['name'] for p in printers]
job_ids = [j['job_id'] for j in jobs]

model = ConcreteModel()
model.P = RangeSet(0, len(printer_ids)-1)
model.J = RangeSet(0, len(jobs)-1)

model.x = Var(model.J, model.P, domain=Binary)
model.s = Var(model.J, domain=NonNegativeReals)
model.Cmax = Var(domain=NonNegativeReals)



def printer_supports_material(printer, required_material):
    if isinstance(required_material, list):
        return any(m['type'] in required_material for m in printer['materials'])
    else:
        return any(m['type'] == required_material for m in printer['materials'])

model.compatible = ConstraintList()
for j in range(len(jobs)):
    compatible = jobs[j]['model']['compatible_printers']
    required_material = jobs[j]['model']['material']
    for p in range(len(printer_ids)):
        printer = printers[p]
        if printer.get('model_name') not in compatible or not printer_supports_material(printer, required_material):
            model.compatible.add(model.x[j, p] == 0)

model.job_assign = ConstraintList()
for j in range(len(jobs)):
    model.job_assign.add(sum(model.x[j,p] for p in range(len(printer_ids))) == 1)

model.no_overlap = ConstraintList()
for p in range(len(printer_ids)):
    for j1 in range(len(jobs)):
        for j2 in range(len(jobs)):
            if j1 < j2:
                M = 1e6
                t1 = jobs[j1]['print_time']
                t2 = jobs[j2]['print_time']
                model.no_overlap.add(
                    model.s[j1] + t1 <= model.s[j2] + M*(2 - model.x[j1,p] - model.x[j2,p])
                )
                model.no_overlap.add(
                    model.s[j2] + t2 <= model.s[j1] + M*(2 - model.x[j1,p] - model.x[j2,p])
                )

model.makespan = ConstraintList()
for j in range(len(jobs)):
    t = jobs[j]['print_time']
    model.makespan.add(model.Cmax >= model.s[j] + t)

model.obj = Objective(expr=model.Cmax, sense=minimize)

for j in range(len(jobs)):
    compatible = jobs[j]['model']['compatible_printers']
    required_material = jobs[j]['model']['material']
    found = False
    for p, printer in enumerate(printers):
        if printer.get('model_name') in compatible:
            if any(m['type'] == required_material or (isinstance(required_material, list) and m['type'] in required_material) for m in printer['materials']):
                found = True
    if not found:
        print(f"WARNING: No compatible printer found for job {j} (model_id={jobs[j]['model_id']}) with compatible_printers={compatible} and required_material={required_material}")

solver = SolverFactory('highs')
solver.config.load_solutions = False
result = solver.solve(model, tee=False)

printer_job_map = {pid: [] for pid in printer_ids}
for j in range(len(jobs)):
    for p in range(len(printer_ids)):
        if value(model.x[j, p]) > 0.5:
            printer_job_map[printer_ids[p]].append(jobs[j]['model_id'])
with open('result.txt', 'w') as f:
    for printer in printer_ids:
        f.write(f"{printer}:\n")
        for job in printer_job_map[printer]:
            f.write(f"  {job}\n")