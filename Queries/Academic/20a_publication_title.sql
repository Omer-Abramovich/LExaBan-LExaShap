SELECT 
    publication.title
FROM 
    author, conference, publication, writes
WHERE
    conference.name = 'VLDB' 
    AND (publication.year < 1995 or publication.year > 2002 ) 
    AND publication.citation_count > 100    
    AND author.aid = writes.aid 
    AND conference.cid = publication.cid 
    AND publication.pid = writes.pid
GROUP BY 
    publication.title