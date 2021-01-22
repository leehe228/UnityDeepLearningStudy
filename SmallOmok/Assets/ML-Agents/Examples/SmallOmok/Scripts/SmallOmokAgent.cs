using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using UnityEngine.UI;

public class SmallOmokAgent : Agent {

    public SmallOmokAgent Opponent;

    public GameObject Stone;

    private SmallOmokAcademy academy;

    List<GameObject> stoneList;

    public Text scoreBoard;

    public bool TURN;
    int lastTemp;

    int count;

    public override void InitializeAgent()
    {
        academy = FindObjectOfType(typeof(SmallOmokAcademy)) as SmallOmokAcademy;

        academy.ClearMap();
        stoneList = new List<GameObject>();

        count = 0;
        lastTemp = 0;

        scoreBoard.text = "START!\n";
    }

    public override void CollectObservations()
    {
        // No Vector Obs
    }

    public override void AgentAction(float[] vectorAction, string textAction)
	{
        //Debug.Log("tag : " + Stone.tag + ", count : " + count + ", TURN : " + TURN);
        if (TURN)
        {
            count++;
            SetReward(-0.01f);
            int temp = (int)(vectorAction[0]);

            int xIndex = temp / 11;
            int zIndex = temp % 11;

            if (lastTemp != temp && temp != 0 && academy.mapTable[xIndex, zIndex] == 0)
            {
                Vector3 placePos = new Vector3(xIndex - 5f, 0.1f, zIndex - 5f);

                stoneList.Add(Instantiate(Stone, placePos, Quaternion.identity, transform) as GameObject);
                lastTemp = temp;

                if (Stone.CompareTag("BlackStone"))
                {
                    academy.mapTable[xIndex, zIndex] = 1;
                }
                else
                {
                    academy.mapTable[xIndex, zIndex] = -1;
                }

                if (FinishCheck(xIndex, zIndex) == 1)
                {
                    scoreBoard.text += Stone.tag + " WIN!\n";
                    //Debug.Log(Stone.tag + " Win!");
                    SetReward(5f);
                    Opponent.SetReward(-1f);
                    Done();
                    AgentReset();
                }
                else
                {
                    //Debug.Log(Stone.tag + " : (" + xIndex + ", " + zIndex + ")");
                    float BonusScore = BlockCheck(xIndex, zIndex);
                    SetReward(BonusScore);
                    Opponent.SetReward(BonusScore * -0.5f);
                    TURN = false;
                    Opponent.TURN = true;
                }
            }
            else
            {
                SetReward(-0.05f);
            }
        }
    }

    public override void AgentReset()
    {
        academy.AcademyReset();
        AgentResetByAcademy();
        //Opponent.AgentResetByAcademy();
}

    public void AgentResetByAcademy()
    {
        count = 0;

        foreach (GameObject o in stoneList)
        {
            Destroy(o);
        }
        /*GameObject[] temp = GameObject.FindGameObjectsWithTag(Stone.tag);
        for(int i = 0; i < temp.Length; i++)
        {
            Destroy(temp[i]);
        }*/
        stoneList = new List<GameObject>();
    }

    public override void AgentOnDone()
    {

    }

    public float BlockCheck(int xIndex, int zIndex)
    {
        int x, z;
        int count;
        float SCORE = 0f;

        x = xIndex;
        z = zIndex;
        count = 0;
        // 가로
        for (int i = Mathf.Max(0, x - 4); i < Mathf.Min(11, x + 5); i++)
        {
            if (academy.mapTable[i, z] != 0)
            {
                count++;
            }
        }
        if (count >= 3) SCORE += 0.1f;

        x = xIndex;
        z = zIndex;
        count = 0;
        // 세로
        for (int j = Mathf.Max(0, z - 4); j < Mathf.Min(11, z + 5); j++)
        {
            if (academy.mapTable[x, j] != 0)
            {
                count++;
            }
        }
        if (count >= 3) SCORE += 0.1f;

        x = xIndex;
        z = zIndex;
        count = 0;
        // 정대각
        for (int i = Mathf.Max(0, x - 4), j = Mathf.Max(0, z - 4); (i < Mathf.Min(11, x + 5) && j < Mathf.Min(11, z + 5)); i++, j++)
        {
            if (academy.mapTable[i, j] != 0)
            {
                count++;
            }
        }
        if (count >= 3) SCORE += 0.1f;

        x = xIndex;
        z = zIndex;
        count = 0;
        // 반대각
        for (int i = Mathf.Max(0, x - 4), j = Mathf.Min(10, z + 4); (i < Mathf.Min(11, x + 5) && j > Mathf.Max(-1, z - 5)); i++, j--)
        {
            if (academy.mapTable[i, j] != 0)
            {
                count++;
            }
        }
        if (count >= 3) SCORE += 0.1f;

        return SCORE;
    }

    public int FinishCheck(int xIndex, int zIndex)
    {
        int x, z;
        int count;
        int STONE;

        if (Stone.CompareTag("BlackStone"))
        {
            STONE = 1;
        }
        else
        {
            STONE = -1;
        }

        x = xIndex;
        z = zIndex;
        count = 0;
        // 가로
        for (int i = Mathf.Max(0, x - 4); i < Mathf.Min(11, x + 5); i++)
        {
            if (academy.mapTable[i, z] == STONE)
            {
                count++;
            }
        }
        if (count == 5) return 1;

        x = xIndex;
        z = zIndex;
        count = 0;
        // 세로
        for (int j = Mathf.Max(0, z - 4); j < Mathf.Min(11, z + 5); j++)
        {
            if (academy.mapTable[x, j] == STONE)
            {
                count++;
            }
        }
        if (count == 5) return 1;

        x = xIndex;
        z = zIndex;
        count = 0;
        // 정대각
        for (int i = Mathf.Max(0, x - 4), j = Mathf.Max(0, z - 4); (i < Mathf.Min(11, x + 5) && j < Mathf.Min(11, z + 5)); i++, j++)
        {
            if (academy.mapTable[i, j] == STONE)
            {
                count++;
            }
        }
        if (count == 5) return 1;

        x = xIndex;
        z = zIndex;
        count = 0;
        // 반대각
        for (int i = Mathf.Max(0, x - 4), j = Mathf.Min(10, z + 4); (i < Mathf.Min(11, x + 5) && j > Mathf.Max(-1, z - 5)); i++, j--)
        {
            if (academy.mapTable[i, j] == STONE)
            {
                count++;
            }
        }
        if (count == 5) return 1;

        return 0;
    }
}
