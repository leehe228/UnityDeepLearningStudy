using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Controller : MonoBehaviour
{
    public int[,] mapTable = new int[11, 11];

    public bool TURN;
    public GameObject BlackStone;
    public GameObject WhiteStone;

    public SmallOmokAgent AgentA, AgentB;

    void Start()
    {
        AgentA = GameObject.Find("AgentA").GetComponent<SmallOmokAgent>();
        AgentB = GameObject.Find("AgentB").GetComponent<SmallOmokAgent>();

        TURN = true;
        for (int i = 0; i < 11; i++)
        {
            for (int j = 0; j < 11; j++)
            {
                mapTable[i, j] = 0;
            }
        }
    }

    void Update()
    {
        
    }

    public void Reset()
    {
        mapTable = new int[11, 11];
    }

    public bool PutStone(int xIndex, int zIndex, GameObject stone)
    {
        Debug.Log(xIndex + ", " + zIndex + " : " + stone.tag);

        Vector3 placePos = new Vector3(xIndex - 9, 0, zIndex - 9);
        if (mapTable[xIndex, zIndex] == 0)
        {
            if (stone.CompareTag("BlackStone"))
            {
                mapTable[xIndex, zIndex] = 1;
                Instantiate(BlackStone, placePos, Quaternion.identity, transform);
                return true;
            }
            else
            {
                mapTable[xIndex, zIndex] = -1;
                Instantiate(WhiteStone, placePos, Quaternion.identity, transform);
                return true;
            }
        }
        else
        {
            return false;
        }
    }

    public void ChangeTurn()
    {
        TURN = !TURN;
    }

    public int FinishCheck(int xIndex, int zIndex)
    {
        return 0;
    }
}
