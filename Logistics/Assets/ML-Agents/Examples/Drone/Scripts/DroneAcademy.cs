using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class DroneAcademy : Academy {

    public GameObject agent1, agent2, agent3;
    public GameObject map;

    public override void AcademyReset()
    {
        agent1.transform.position = new Vector3(0f, 5f, 0f);
        agent2.transform.position = new Vector3(0f, 6f, 0f);
        agent3.transform.position = new Vector3(0f, 7f, 0f);

        map.GetComponent<MapController>().generate1();
        map.GetComponent<MapController>().generate2();
        // map.GetComponent<MapController>().generate3();
    }

    public override void AcademyStep()
    {
        
    }
}
