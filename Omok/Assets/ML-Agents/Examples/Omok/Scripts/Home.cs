using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Home : MonoBehaviour
{
    public GameObject BlackStone;
    public GameObject WhiteStone;

    public Vector3 MousePos;

    //public OmokAgent agentA, agentB;

    public GameObject marker;

    public bool TURN;

    // 바둑판
    public int[,] mapTable = new int[19, 19];

    private Camera cam;

    void Start()
    {
        //agentA = GameObject.Find("OmokAgentA").GetComponent<OmokAgent>();
        //agentB = GameObject.Find("OmokAgentB").GetComponent<OmokAgent>();
        cam = Camera.main;
        TURN = true;
    }

    void Update()
    {
        if (TURN)
        {
            PlayerPlay();
        }
    }

    public void PlayerPlay()
    {
        int xIndex, zIndex;

        // 바둑알 놓기
        Vector3 mos = Input.mousePosition;
        mos.z = cam.farClipPlane;
        Vector3 dir = cam.ScreenToWorldPoint(mos);
        RaycastHit hit;

        if (Physics.Raycast(transform.position, dir, out hit, mos.z))
        {
            Vector3 tempPos = new Vector3(
                Mathf.Round(hit.point.x),
                Mathf.Round(hit.point.y),
                Mathf.Round(hit.point.z));

            xIndex = (int)(tempPos.x) + 9;
            zIndex = (int)(tempPos.z) + 9;

            if (0 <= xIndex && xIndex <= 18 && 0 <= zIndex && zIndex <= 18)
            {
                marker.transform.position = tempPos;

                if (Input.GetMouseButton(0) && mapTable[xIndex, zIndex] != 1 && mapTable[xIndex, zIndex] != -1)
                {
                    PutStone(xIndex, zIndex);
                }
            }
        }
    }

    public void PutStone(int xIndex, int zIndex)
    {
        Vector3 tempPos = new Vector3(xIndex - 9, 0, zIndex - 9);

        if (TURN)
        {
            Instantiate(BlackStone, tempPos, Quaternion.identity, transform);
            mapTable[xIndex, zIndex] = 1;
        }
        else
        {
            Instantiate(WhiteStone, tempPos, Quaternion.identity, transform);
            mapTable[xIndex, zIndex] = -1;
        }

        int f = FinishCheck(xIndex, zIndex);
        
        if (f == 1)
        {
            if (TURN)
            {
                Debug.Log("A Win!");
                //agentA.SetWinReward();
                //agentB.SetLoseReward();
            }
            else
            {
                Debug.Log("B Win!");
                //agentA.SetLoseReward();
                //agentB.SetWinReward();
            }
            
        }

        ChangeTurn();
    }

    public void ChangeTurn()
    {
        TURN = true;
    }

    public void Reset()
    {
        Destroy(BlackStone);
        Destroy(WhiteStone);
        mapTable = new int[19, 19];
    }

    public int FinishCheck(int xIndex, int zIndex)
    {
        int stone = mapTable[xIndex, zIndex];

        int count;
        int x, z;

        // 가로 <-
        x = xIndex;
        z = zIndex;
        count = 0;
        while(true) 
        {
            if (count == 5) return 1;
            else if (x == 18) break;
            if (mapTable[x, z] == stone)
            {
                x++;
                count++;
            }
            else break;
        }
        // 가로 ->
        x = xIndex;
        z = zIndex;
        count = 0;
        while (true)
        {
            if (count == 5) return 1;
            else if (x == 0) break;
            if (mapTable[x, z] == stone)
            {
                x--;
                count++;
            }
            else break;
        }

        // 세로 위
        x = xIndex;
        z = zIndex;
        count = 0;
        while (true)
        {
            if (count == 5) return 1;
            else if (z == 18) break;
            if (mapTable[x, z] == stone)
            {
                z++;
                count++;
            }
            else break;
        }
        // 세로 아래
        x = xIndex;
        z = zIndex;
        count = 0;
        while (true)
        {
            if (count == 5) return 1;
            else if (z == 0) break;
            if (mapTable[x, z] == stone)
            {
                z--;
                count++;
            }
            else break;
        }

        // 대각선 왼쪽위
        x = xIndex;
        z = zIndex;
        count = 0;
        while (true)
        {
            if (count == 5) return 1;
            else if (x == 0 || z == 18) break;
            if (mapTable[x, z] == stone)
            {
                x--;
                z++;
                count++;
            }
            else break;
        }
        // 대각선 왼쪽아래
        x = xIndex;
        z = zIndex;
        count = 0;
        while (true)
        {
            if (count == 5) return 1;
            else if (x == 0 || z == 0) break;
            if (mapTable[x, z] == stone)
            {
                x--;
                z--;
                count++;
            }
            else break;
        }

        // 대각선 오른쪽 위
        x = xIndex;
        z = zIndex;
        count = 0;
        while (true)
        {
            if (count == 5) return 1;
            else if (z == 18 || z == 18) break;
            if (mapTable[x, z] == stone)
            {
                x++;
                z++;
                count++;
            }
            else break;
        }
        // 대각선 오른쪽 아래
        x = xIndex;
        z = zIndex;
        count = 0;
        while (true)
        {
            if (count == 5) return 1;
            else if (x == 18 || z == 0) break;
            if (mapTable[x, z] == stone)
            {
                x++;
                z--;
                count++;
            }
            else break;
        }

        return 0;
    }
}
