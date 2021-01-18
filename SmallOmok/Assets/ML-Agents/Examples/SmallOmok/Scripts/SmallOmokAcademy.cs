using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class SmallOmokAcademy : Academy {

    public SmallOmokAgent AgentA;
    public SmallOmokAgent AgentB;

    public int[,] mapTable = new int[11, 11];

    public override void AcademyReset()
    {
        AgentA.TURN = false;
        AgentB.TURN = false;
        ClearMap();
    }

    public void ClearMap()
    {
        for (int i = 0; i < 11; i++)
        {
            for (int j = 0; j < 11; j++)
            {
                mapTable[i, j] = 0;
            }
        }
        GameObject[] temp = GameObject.FindGameObjectsWithTag("BlackStone");
        for (int i = 0; i < temp.Length; i++)
        {
            Destroy(temp[i]);
        }

        temp = GameObject.FindGameObjectsWithTag("WhiteStone");
        for (int i = 0; i < temp.Length; i++)
        {
            Destroy(temp[i]);
        }

        AgentA.TURN = true;
        AgentB.TURN = false;
    }

    public override void AcademyStep()
    {


    }

}
