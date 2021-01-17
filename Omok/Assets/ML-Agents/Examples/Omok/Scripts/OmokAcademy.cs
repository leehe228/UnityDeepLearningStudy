using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class OmokAcademy : Academy {

    public Home home;

    public override void AcademyReset()
    {
        home = GameObject.Find("Main Camera").GetComponent<Home>();
        home.Reset();
    }

    public override void AcademyStep()
    {


    }

}
