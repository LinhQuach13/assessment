# Technical Assessment

## Hypothetical python library named "cms_procedures":
**get_procedure_attributes(procedure_id = None)**
- Created this library assuming values are known for attributes. 2 values for each key was chosen for initial work. An if statement was added to account for if procedure id is not specified it would randomly generate a number for procedure that is less than 5000.
- Procedure_id: Unique identifier for each procedure.
- Procedure type: This is the type of procedure performed and values are strings because it is assumed most procedures will have their names in string.
- Duration: This column is for how long the procedure lasted and values are numeric datatype (float). The time is in assumed to be in minutes.
- Severity: This is the severity of the condition being addressed. The measurement is a numerica datatype (float) ranging from 1 (asymptomatic) to 5 (catastrophic manifestations).

**get_procedure_success(procedure_id)**
- In this function the target(success/failure) would be assumed to be  numerical. During the data preparation phase of the pipeline I would have converted success to the the number 1 and failure to the number 0.

**get_procedure_outcomes(procedure_id)**
- Created this library assuming values are known for attributes. 2 values for each key was chosen for initial work. 
- Procedure_id: Unique identifier for each procedure.
- Severity of post-op procedure complications: Range is 1 (minor) to 4 (mortality).
- Pain: amount of pain range from 0 (none) to 10 (severe). This value is in a numeric datatype (float)
- Recurrence: recurrence of original condition and this measurement is assumed to be in percentages. The value is a numeric datatype (float).