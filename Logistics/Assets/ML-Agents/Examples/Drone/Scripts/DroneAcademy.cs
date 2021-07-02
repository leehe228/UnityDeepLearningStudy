using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class DroneAcademy : Academy {

    public GameObject hub;
    public GameObject box1, box2, box3;
    public GameObject destination;

    public override void AcademyReset()
    {
        box1 = GameObject.FindWithTag("box1");
        box2 = GameObject.FindWithTag("box2");
        box3 = GameObject.FindWithTag("box3");
        hub = GameObject.FindWithTag("hub");
        destination = GameObject.FindWithTag("destination");

    }

    public override void AcademyStep()
    {


    }

}
