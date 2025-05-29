SELECT 
    publication.year
FROM 
    publication, conference
WHERE
    publication.title LIKE 'Making %'
    AND publication.cid = conference.cid
    AND conference.name = 'VLDB'
GROUP BY 
    publication.year