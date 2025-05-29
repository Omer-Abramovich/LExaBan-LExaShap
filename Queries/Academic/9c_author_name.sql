SELECT 
    author.name
FROM 
    author, conference, publication, writes, organization
WHERE
    author.aid = writes.aid 
    AND writes.pid = publication.pid    
    AND publication.cid = conference.cid 
    AND author.oid = organization.oid 
    AND conference.name = 'SIGMOD' 
    AND organization.name = 'University of Michigan'
    AND publication.year = 2005    
GROUP BY author.name