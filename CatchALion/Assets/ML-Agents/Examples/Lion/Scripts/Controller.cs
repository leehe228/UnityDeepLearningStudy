using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Controller : MonoBehaviour
{
    public int[,] mapTable;

    private const int LION = 1;
    private const int GIRAFFE = 2;
    private const int ELEPHANT = 3;
    private const int CHICK = 4;
    private const int CHICKEN = 5;
    private const int EMPTY = 0;

    public GameObject LionStone;
    public GameObject GiraffeStone;
    public GameObject ElephantStone;
    public GameObject ChickStone;
    public GameObject ChickenStone;

    public GameObject marker;
    public GameObject LastMarker;
    public List<GameObject> markerList;

    private Camera cam;

    private int LastX, LastZ;
    private int LastTouched;

    List<Vector3> markerVectorList;
    List<GameObject> StoneList;
    GameObject[] animalList;

    void Start()
    {
        mapTable = new int[4, 3] { { GIRAFFE, LION, ELEPHANT }, { EMPTY, CHICK, EMPTY }, { EMPTY, EMPTY, EMPTY }, { EMPTY, EMPTY, EMPTY } };
        cam = Camera.main;
        SetStones();
        markerList = new List<GameObject>();
        markerVectorList = new List<Vector3>();
        StoneList = new List<GameObject>();
    }

    void Update()
    {
        PlayerPlay();
    }

    public void PlayerPlay()
    {
        int xIndex, zIndex;

        Vector3 mos = Input.mousePosition;
        mos.z = cam.farClipPlane;
        Vector3 dir = cam.ScreenToWorldPoint(mos);
        RaycastHit hit;

        if (Physics.Raycast(transform.position, dir, out hit, mos.z))
        {
            Vector3 tempPos = new Vector3(
                Mathf.Round(hit.point.x + 1f),
                Mathf.Round(hit.point.y),
                Mathf.Round(hit.point.z + 1.5f));

            xIndex = (int)(tempPos.x);
            zIndex = (int)(tempPos.z);

            if (0 <= xIndex && xIndex < 3 && 0 <= zIndex && zIndex < 4)
            {
                // marker.transform.position = tempPos;

                int count = 0;

                if (Input.GetMouseButton(0))
                {
                    if (markerVectorList != null && markerVectorList.Count != 0)
                    {
                        foreach (Vector3 v in markerVectorList)
                        {
                            if (tempPos.x - 1.5f <= v.x && v.x <= tempPos.x - 0.5f && tempPos.z - 2f <= v.z && v.z <= tempPos.z - 1f)
                            {
                                count++;
                            }
                        }
                        if (count != 0)
                        {
                            MoveTo(xIndex, zIndex);
                        }
                        else
                        {
                            Touched(xIndex, zIndex);
                        }
                    }
                    else
                    {
                        Touched(xIndex, zIndex);
                    }
                }
            }
            else
            {
                //Debug.Log("Outside of the Map");
            }
        }
    }

    public void Touched(int xIndex, int zIndex)
    {
        if (markerList.Count > 0)
        {
            foreach (GameObject m in markerList)
            {
                Destroy(m);
            }
        }
        markerList.Clear();

        int Animal = mapTable[zIndex, xIndex];
        switch (Animal)
        {
            case LION:
                {
                    Debug.Log("Lion");
                    break;
                }
            case GIRAFFE:
                {
                    Debug.Log("Giraffe");
                    break;
                }
            case ELEPHANT:
                {
                    Debug.Log("Elephant");
                    break;
                }
            case CHICK:
                {
                    Debug.Log("Chick");
                    break;
                }
            case CHICKEN:
                {
                    Debug.Log("Chicken");
                    break;
                }
            case EMPTY:
                {
                    Debug.Log("EMPTY");
                    break;
                }
        }
        SetMarker(xIndex, zIndex, Animal);

        LastX = xIndex;
        LastZ = zIndex;
        LastTouched = Animal;
    }

    private void MoveTo(int xIndex, int zIndex)
    {
        for(int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (mapTable[i, j] == LastTouched)
                {
                    mapTable[i, j] = 0;
                    mapTable[xIndex, zIndex] = LastTouched;
                    break;
                }
            }
        }
        SetStones();
    }

    private void DrawMarker()
    {
        foreach (Vector3 v in markerVectorList)
        {
            Debug.Log("markerVectorList : " + v);
            if (0f <= v.x + 1f && v.x + 1f <= 2f && 0f <= v.z + 1.5f && v.z + 1.5f <= 3f)
            {
                markerList.Add(Instantiate(marker, v, Quaternion.identity, transform) as GameObject);
            }
        }
    }

    private void SetMarker(int xIndex, int zIndex, int Animal)
    {
        markerVectorList.Clear();

        float centerX = xIndex - 1f;
        float centerZ = zIndex - 1.5f; 

        switch (Animal)
        {
            case CHICK:
                {
                    markerVectorList.Add(new Vector3(centerX, 0.2f, centerZ + 1));
                    break;
                }
            case GIRAFFE:
                {
                    markerVectorList.Add(new Vector3(centerX, 0.2f, centerZ + 1));
                    markerVectorList.Add(new Vector3(centerX, 0.2f, centerZ - 1));
                    markerVectorList.Add(new Vector3(centerX + 1, 0.2f, centerZ));
                    markerVectorList.Add(new Vector3(centerX - 1, 0.2f, centerZ));
                    break;
                }
            case LION:
                {
                    markerVectorList.Add(new Vector3(centerX, 0.2f, centerZ + 1));
                    markerVectorList.Add(new Vector3(centerX, 0.2f, centerZ - 1));
                    markerVectorList.Add(new Vector3(centerX + 1, 0.2f, centerZ));
                    markerVectorList.Add(new Vector3(centerX - 1, 0.2f, centerZ));
                    markerVectorList.Add(new Vector3(centerX + 1, 0.2f, centerZ + 1));
                    markerVectorList.Add(new Vector3(centerX - 1, 0.2f, centerZ + 1));
                    markerVectorList.Add(new Vector3(centerX + 1, 0.2f, centerZ - 1));
                    markerVectorList.Add(new Vector3(centerX - 1, 0.2f, centerZ - 1));
                    break;
                }
            case ELEPHANT:
                {
                    markerVectorList.Add(new Vector3(centerX + 1, 0.2f, centerZ + 1));
                    markerVectorList.Add(new Vector3(centerX - 1, 0.2f, centerZ + 1));
                    markerVectorList.Add(new Vector3(centerX + 1, 0.2f, centerZ - 1));
                    markerVectorList.Add(new Vector3(centerX - 1, 0.2f, centerZ - 1));
                    break;
                }
        }

        DrawMarker();
    }

    public void SetStones()
    {
        if(StoneList != null && StoneList.Count != 0)
        {
            foreach (GameObject o in StoneList)
            {
                Destroy(o);
            }
            StoneList.Clear();
        }

        for(int i = 0; i < 4; i++)
        {
            for(int j = 0; j < 3; j++)
            {
                Vector3 pos = new Vector3((float)j - 1f, 0.1f, (float)i - 1.5f);
                switch(mapTable[i, j])
                {
                    case LION:
                        {
                            StoneList.Add(Instantiate(LionStone, pos, Quaternion.identity, transform) as GameObject);
                            break;
                        }
                    case GIRAFFE:
                        {
                            StoneList.Add(Instantiate(GiraffeStone, pos, Quaternion.identity, transform) as GameObject);
                            break;
                        }
                    case ELEPHANT:
                        {
                            StoneList.Add(Instantiate(ElephantStone, pos, Quaternion.identity, transform) as GameObject);
                            break;
                        }
                    case CHICK:
                        {
                            StoneList.Add(Instantiate(ChickStone, pos, Quaternion.identity, transform) as GameObject);
                            break;
                        }
                    case CHICKEN:
                        {
                            StoneList.Add(Instantiate(ChickenStone, pos, Quaternion.identity, transform) as GameObject);
                            break;
                        }
                }
            }
        }
    }

}
