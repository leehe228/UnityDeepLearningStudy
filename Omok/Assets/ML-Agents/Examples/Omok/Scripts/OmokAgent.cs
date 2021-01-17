using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

public class OmokAgent : Agent {

    public Home home;

    public override void InitializeAgent()
    {
        home = GameObject.Find("Main Camera").GetComponent<Home>();
    }

    public override void CollectObservations()
    {
        
    }

    public override void AgentAction(float[] vectorAction, string textAction)
	{
        int xIndex = (int)(Mathf.Clamp(vectorAction[0], 0f, 18f));
        int zIndex = (int)(Mathf.Clamp(vectorAction[1], 0f, 18f));

        home.PutStone(xIndex, zIndex);
    }

    public override void AgentReset()
    {

    }

    public override void AgentOnDone()
    {

    }

    public void SetLoseReward()
    {
        SetReward(-1);
        Done();
    }

    public void SetWinReward()
    {
        SetReward(1);
        Done();
    }

    public void SetBlockedReward()
    {
        SetReward(-0.2f);
    }

    public void SetBlockReward()
    {
        SetReward(0.2f);
    }
}
